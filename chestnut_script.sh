#!/bin/bash

#####################################################
#SBATCH --partition=RM-shared 
####SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sskemkar@seas.upenn.edu
#SBATCH --output=output.dat         # Standard output file
#SBATCH --error=stderr_file         # Standard error file
#####################################################
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR >job_details.dat
echo "starting time: ">>job_details.dat
date >>job_details.dat
module load CP2K/8.1-gcc10.2.0-openmpi4.0.5
jobid=`echo $SLURM_JOBID | cut -f1 -d '.'` 
rundir=/scratch/job-$jobid 
echo "name of rundir is $rundir" >>job_details.dat 
###mkdir -pv $rundir
###cp -r ./* $rundir
###cd $rundir
./project ./config/PhysiCell_settings.xml
#####export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:./addons/libRoadrunner/roadrunner/lib
####./ode_energy
###mv * $SLURM_SUBMIT_DIR
###cd /tmp
###rmdir $rundir

