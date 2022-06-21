# 3D-pose-estimation-for-ergonomics-analysis

_Marchello G, Abidi H, Farajtabar M, Lahoud M, Fontana E, D’Imperio M, Cannella F_

Sedentary lifestyle is widely known as a cause of different musculoskeletal disorders. The effects of sitting all day for work can however be mitigated by correcting the postures. Monitoring human pose to perform ergonomics analysis may help reporting wrong or potentially harmful postures, and consequently preventing health problems. In this paper, we propose a vision-based system tracking in real-time the pose of manufacturing operators in 3D. The pose estimation is performed by applying OpenPose [1] to the RGB images recorded by the depth camera RealSense d435i, and matching the results to their corresponding depth values. The angles of the joints composing the reconstructed skeleton are then computed, and are further used to evaluate the associated discomfort levels. Such values represent the deviations from optimal working limits, as suggested by current ergonomic studies. This work is part of the SOFTMANBOT project, a cross-sectorial project funded by the EU Horizon 2020 framework, automating textile production lines. Consequently, the project aims at increasing the productivity, yet reducing the health risks of the workers.

# Method

Hey! This is a clone of the tf-pose-estimation by Ildoo Kim modified to work with Tensorflow 2.0+!
Link to original repo: https://www.github.com/ildoonet/tf-openpose

## How to cite
```
@article{,
  title={3D pose estimation for ergonomics analysis of manufactory labours},
  author={Marchello, Gabriele and Abidi, Haider and Farajtabar, Mohammad and Lahoud, Marcel and Fontana, Eleonora and D'Imperio, Mariapaola and Cannella, Ferdinando},
  journal={},
  year={2022},
  publisher={}
}
```

### References 

[1] Cao, Zhe, et al. "Realtime multi-person 2d pose estimation using part affinity fields." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017
