## Set up
* Clone the repository.

* Make conda env (If you don't want to use virtual env, you may skip this)
```
source ./Script/0_conda_env_setup.sh
```

* install pytorch and pytorch lightening. You must check your CUDA Version and install compatible version.
```
source ./Script/1_pytorch_install.sh
```

* setup this repository
```
source ./Script/2_setup.sh
```

* clone torch-jaekwon
```
git clone https://github.com/jakeoneijk/TorchJaekwon.git
```

* edit the *TORCH_JAEKWON_PATH* in *./Script/3_install_torchjk.sh*

* install torch-jaekwon
```
source ./Script/3_install_torchjk.sh
```

* download pretrained module
[download link](http://www.google.co.kr)

* move it to the right place
```
.
└── AudioLDMControlNetInfer
    └── ModelWeight
```

## Use
* check Test.py