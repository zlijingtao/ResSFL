#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=resnet20
batch_size=128

num_agent=2
num_epochs=200
dataset_list="cifar10"
scheme=V2_epoch
random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

regularization=gan_adv_step3
learning_rate=0.005
local_lr=-1
regularization_strength_list="0.4"
cutlayer_list="4"
num_agent_list="1"
ssim_threshold=0.5
train_gan_AE_type=res_normN4C64
bc_list="2"
gan_loss_type=SSIM
folder_name=new_saves/resnet20
for bc in $bc_list; do
        bottleneck_option=norelu_C${bc}S1
        for dataset in $dataset_list; do
                for random_seed in $random_seed_list; do
                        for regularization_strength in $regularization_strength_list; do
                                for cutlayer in $cutlayer_list; do
                                        for num_agent in $num_agent_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_agent}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}_ssim_${ssim_threshold}
                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                                        --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                        --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold}

                                                target_agent=0
                                                attack_scheme=MIA
                                                attack_epochs=50
                                                average_time=1

                                                internal_C=16
                                                N=0
                                                test_gan_AE_type=conv_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_agent=${target_agent} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}

                                                internal_C=16
                                                N=0
                                                test_gan_AE_type=res_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_agent=${target_agent} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                
                                                
                                                internal_C=32
                                                N=2
                                                test_gan_AE_type=res_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_agent=${target_agent} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                

                                                internal_C=64
                                                N=4
                                                test_gan_AE_type=res_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_agent=${target_agent} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                
                                                pretrain_dir=pretrained_models/${arch}_cutlayer_${cutlayer}_bottleneck_${bottleneck_option}_dataset_${dataset}
                                                mkdir ${pretrain_dir}
                                                save_dir=${folder_name}/${filename}
                                                cp $save_dir/checkpoint_classifier_best.tar ${pretrain_dir}/
                                                cp $save_dir/checkpoint_cloud_best.tar ${pretrain_dir}/
                                                cp $save_dir/checkpoint_f_best.tar ${pretrain_dir}/
                                                cp $save_dir/MIA.log ${pretrain_dir}/

                                        done
                                done
                        done
                done
        done
done