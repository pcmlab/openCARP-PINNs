
#$ -j y
#$ -pe smp 8          # 8 cores per GPU
#$ -l h_rt=72:0:0   # 240 hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -l gpu=1           # request 1 GPU
#$ -l cluster=andrena # use the Andrena nodes

module load miniforge
mamba activate env_3.8
python /data/home/exx975/O-PINNs/O-PINNs/main.py /data/home/exx975/O-PINNs/O-PINNs/config.yaml AP_Pl_25June_005diffdata
#python /data/home/exx975/PINNS/main.py -m 3_Apr -vf /data/home/exx975/PINNS/Data/Double_Corner/vm.igb -wf /data/home/exx975/PINNS/Data/Double_Corner/V.igb -ptf /data/home/exx975/PINNS/Data/Mesh/Square_i -a -p
