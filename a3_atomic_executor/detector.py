"""On-demand coarse/fine perception for this known upright cookie geometry.

Inputs are RGB-D, calibration, known dimensions and an optional caller ROI.
No simulator, object world coordinates, count, row pitch, labels or old pose.
The appearance mask is a yellow-cookie scene baseline, not semantic recognition.
Fine poses assume upright pieces on one approximately level support plane.
"""
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class CookieObservation:
    mode: str
    center_world: np.ndarray | None
    bounds_world: np.ndarray | None
    poses_world: list
    pixel_thickness: float
    elapsed_ms: float
    labels: np.ndarray
    mask: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class AdaptiveCookieRGBD:
    def __init__(self, full_size_m=(.050, .0095/1.5, .025), *, fine_threshold_px=8):
        self.size=np.asarray(full_size_m,dtype=float)
        self.fine_threshold_px=float(fine_threshold_px)

    def estimate(self,rgb,depth_m,K,camera_to_world,*,detail='auto',roi=None):
        start=time.perf_counter()
        rgb=np.asarray(rgb)
        depth=np.asarray(depth_m,dtype=np.float32)
        K=np.asarray(K,dtype=float)
        T=np.asarray(camera_to_world,dtype=float)
        if detail not in ('auto','coarse','fine'):
            raise ValueError('detail must be auto, coarse or fine')
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=(hsv[...,0]>14)&(hsv[...,0]<38)&(hsv[...,1]>170)&(hsv[...,2]>70)
        mask &= np.isfinite(depth)&(depth>.05)&(depth<2)
        if roi is not None:
            x0,y0,x1,y1=map(int,roi)
            caller_mask=np.zeros(mask.shape,bool)
            caller_mask[max(y0,0):y1,max(x0,0):x1]=True
            mask &= caller_mask
        labels=np.zeros(depth.shape,np.int32)
        v,u=np.nonzero(mask)
        if len(u)<20:
            return CookieObservation('unobserved',None,None,[],0,(time.perf_counter()-start)*1000,labels,mask,
                                     {'reason':'insufficient current-frame yellow depth support'})
        median_depth=cv2.medianBlur(depth,5)
        depth_noise=float(np.median(np.abs(depth[v,u]-median_depth[v,u])))
        denoised=depth_noise>.00040
        if denoised:
            depth=cv2.bilateralFilter(median_depth,5,.003,1.5)
        dd=depth[v,u]
        xyz=np.c_[(u-K[0,2])*dd/K[0,0],(v-K[1,2])*dd/K[1,1],dd]@T[:3,:3].T+T[:3,3]
        lo,hi=np.percentile(xyz,[1,99],axis=0)
        center=(lo+hi)/2
        pixel_thickness=float(min(K[0,0],K[1,1])*self.size[1]/np.median(dd))
        mode=('fine' if pixel_thickness>=self.fine_threshold_px else 'coarse') if detail=='auto' else detail
        poses=[]
        diagnostics={'appearance':'fixed yellow HSV prior','assumption':'upright, common approximate top plane',
                     'masked_points':len(u),'count_is_visible_candidates_only':True,
                     'depth_roughness_m':depth_noise,'current_frame_denoising':denoised}
        if mode=='fine':
            # The top plane is a histogram mode, not a ground-truth table height.
            zmin,zmax=np.percentile(xyz[:,2],[2,99])
            if zmax-zmin>.15:
                diagnostics['reason']='multiple incompatible support heights'
            else:
                edges=np.arange(zmin,zmax+.0002,.0002)
                if len(edges)<3:edges=np.linspace(zmin-.0002,zmax+.0002,5)
                hist,_=np.histogram(xyz[:,2],bins=edges)
                smooth=gaussian_filter1d(hist.astype(float),.8)
                # Prefer an upper support plane. A similarly colored lower box
                # rim can otherwise dominate the global height mode and bias Z.
                upper=np.flatnonzero((edges[:-1]+edges[1:])/2>=zmax-.0035)
                peak=upper[np.argmax(smooth[upper])] if len(upper) else np.argmax(smooth)
                top=float((edges[peak]+edges[peak+1])/2)
                seeds=np.zeros(mask.shape,np.uint8)
                seeds[v,u]=(np.abs(xyz[:,2]-top)<(.0010 if denoised else .0006))
                if denoised:
                    seeds=cv2.morphologyEx(seeds,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
                    seeds[~mask]=0
                # Cut color discontinuities between same-height neighboring pieces.
                hue=(cv2.medianBlur(hsv[...,0],3) if denoised else hsv[...,0]).astype(float)
                grad=np.zeros(depth.shape)
                for axis in (0,1):
                    delta=np.abs(np.diff(hue,axis=axis))
                    if axis==0:
                        grad[1:]=np.maximum(grad[1:],delta)
                        grad[:-1]=np.maximum(grad[:-1],delta)
                    else:
                        grad[:,1:]=np.maximum(grad[:,1:],delta)
                        grad[:,:-1]=np.maximum(grad[:,:-1],delta)
                seeds[grad>2]=0
                # Diagonal one-pixel contacts must not merge neighboring strips.
                count,components,stats,_=cv2.connectedComponentsWithStats(seeds,connectivity=4)
                xyzmap=np.zeros((*depth.shape,3))
                xyzmap[v,u]=xyz
                rejected=0
                for label in range(1,count):
                    if stats[label,4]<10:continue
                    x,y,w,h,_=stats[label]
                    part=components[y:y+h,x:x+w]==label
                    if x==0 or y==0 or x+w==depth.shape[1] or y+h==depth.shape[0]:
                        rejected+=1
                        continue
                    pts=xyzmap[y:y+h,x:x+w][part]
                    _,vectors=np.linalg.eigh(np.cov(pts[:,:2].T))
                    axis=vectors[:,-1]
                    if axis[0]<0:axis=-axis
                    other=np.array([-axis[1],axis[0]])
                    basis=np.c_[axis,other]
                    coords=pts[:,:2]@basis
                    low,high=np.percentile(coords,[1,99],axis=0)
                    span=high-low
                    if not(.94*self.size[0]<span[0]<1.08*self.size[0] and span[1]<(.0045 if denoised else .003)):
                        rejected+=1
                        continue
                    xy=basis@((low+high)/2)
                    p=np.r_[xy,np.median(pts[:,2])-self.size[2]/2]
                    if any(np.linalg.norm(p-old[:3,3])<self.size[1]*.65 for old in poses):
                        rejected+=1
                        continue
                    pose=np.eye(4)
                    pose[:2,:2]=basis
                    pose[:3,3]=p
                    poses.append(pose)
                    labels[y:y+h,x:x+w][part]=len(poses)
                diagnostics.update(top_plane_z_m=top,rejected_components=rejected)
        return CookieObservation(mode,center,np.array([lo,hi]),poses,pixel_thickness,
                                 (time.perf_counter()-start)*1000,labels,mask,diagnostics)
