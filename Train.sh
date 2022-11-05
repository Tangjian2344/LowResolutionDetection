#python Main.py --alpha 1.0 --beta 1.0


# ----------------------------------------------
# 单独训练阶段和之前一致，所以不再重复训练
#python Main.py --alpha 0.01 --beta 1.0 \
#               --pass1 \
#               --weight_sr_fined  ./exped/output_2022_04_05_16_30_26_1.0_1.0/lastest_p1.pth

# 联合训练时被打断现在resume 训练
#python Main.py --alpha 0.01 --beta 1.0 \
#               --resume \
#               --checkpoint  ./exp/output_2022_04_06_09_46_02_0.01_1.0/lastest.pth
# ----------------------------------------------

# -----------------------------------------------------------------------------------------
#python Main.py --alpha 0.00001  --beta 1.0        --save_dir ./exp/output_test
#python Main.py --alpha 0.001    --beta 1.0        --save_dir ./exp/output_test
#python Main.py --alpha 0.1      --beta 1.0        --save_dir ./exp/output_test
#python Main.py --alpha 1.       --beta 1.0        --save_dir ./exp/output_test
#python Main.py --alpha 1.       --beta 0.1        --save_dir ./exp/output_test
#python Main.py --alpha 1.       --beta 0.001      --save_dir ./exp/output_test
#python Main.py --alpha 1.       --beta 0.00001    --save_dir ./exp/output_test
#
#
#python Main.py --alpha 0.00001  --beta 1.0       --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 0.001    --beta 1.0       --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 0.1      --beta 1.0       --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 1.       --beta 1.0       --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 1.       --beta 0.1       --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 1.       --beta 0.001     --save_dir ./exp/output_test  --use_hr
#python Main.py --alpha 1.       --beta 0.00001   --save_dir ./exp/output_test  --use_hr


#python Main.py  --beta 1.      --save_dir ./exp/output_test  --epochs  0 50  --only_det # 仅仅训练d网络
#python Main.py --alpha 1.       --beta 0.01       --save_dir ./exp/output_test  --use_hr  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.05       --save_dir ./exp/output_test  --use_hr  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.5        --save_dir ./exp/output_test  --use_hr  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#
#python Main.py --alpha 1.       --beta 0.01       --save_dir ./exp/output_test  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.05       --save_dir ./exp/output_test  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.5        --save_dir ./exp/output_test  --pass1 --weight_sr_fined ./exp/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# # lr_img 做边界填充 参与D网络的训练

#python Main.py --alpha 0.00001  --beta 1.0        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 0.001    --beta 1.0        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 0.1      --beta 1.0        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 1.0        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.5        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.1        --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.05       --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.01       --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.001      --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.00001    --save_dir ./exp1920/output_test  --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920/output_test_2022_04_08_00_34_22_1e-05_1.0/lastest_p1.pth



# 整个数据集
#python Main.py --alpha 1.   --beta 0.  --save_dir ./exp/output  --epochs  50 0   # just finetued Sr net
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./exp/output  --use_hr  --epochs  0 50  #  combine train directly
#python Main.py --alpha 0.   --beta 1.    --save_dir ./exp/output  --use_hr  --epochs  0 50  # only train with  Det loss
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./exp/output  --use_hr  --epochs  25 25 #  combine train after finetues sr


# 给 lr 停驾高斯模糊  7x7
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./exp/output  --use_hr  --epochs  25 25 #  combine train after finetues sr

# lr 高斯噪声 var=0.004
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./exp/output  --use_hr  --epochs  25 25 #  combine train after finetues sr


# ---------------1920数据集------------------
#python Main.py --beta 1.                         --save_dir ./exp1920new/output_test  --epochs  0 50  --only_det
#python Main.py --alpha 0.00001  --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.0001   --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.001    --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.01     --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.1      --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 1.0       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.1       --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.01      --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.001     --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.0001    --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.00001   --save_dir ./exp1920new/output_test  --use_hr  --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth

#python Main.py --alpha 0.00001  --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.0001   --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.001    --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.01     --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.1      --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 1.0       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.1       --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.01      --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.001     --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.0001    --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.00001   --save_dir ./exp1920new_only_sr/output_test   --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth

#python Main.py --alpha 0.00001  --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.0001   --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.001    --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.01     --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 0.1      --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 1.0     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.1     --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.01    --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.001   --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.0001  --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth
#python Main.py --alpha 1.       --beta 0.00001 --save_dir ./exp1920new_lr/output_test --use_hr --use_lr --pass1 --weight_sr_fined ./exp1920new/lastest_p1.pth

# 整个数据集
#python Main.py --beta 1.                 --save_dir ./expnew/output  --epochs  0 50  --only_det
python Main.py --alpha 1.   --beta 0.1   --save_dir ./expnew/output  --use_hr  --epochs  25 25 #  combine train after finetues sr
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./expnew/output  --use_hr  --epochs  0 50  #  combine train directly
#python Main.py --beta 1.                 --save_dir ./expnew/output  --epochs  0 50  --only_det --weight_od ./YOLOLITE/model/v5lite-s.pth

#python Main.py --alpha 0.   --beta 1.    --save_dir ./expnew/output  --use_hr  --epochs  0 50  # only train with  Det loss
#python Main.py --alpha 1.   --beta 0.    --save_dir ./expnew/output  --epochs  50 0   # just finetued Sr net

# 模糊
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./expnew/output  --use_hr  --epochs  25 25  --degra blur #  combine train after finetues sr

# lr 高斯噪声 var=0.004
#python Main.py --alpha 1.   --beta 0.1   --save_dir ./expnew/output  --use_hr  --epochs  25 25  --degra noise # combine train after finetues sr