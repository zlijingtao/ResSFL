#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128

num_agent=2
num_epochs=200
dataset_list="cifar100 svhn mnist facescrub"
scheme=V2_epoch
random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

regularization=None
scheme_list="V2_epoch"
learning_rate=0.05
local_lr_list="0.0"
ssim_threshold=0.5
regularization_strength_list="0.0"
folder_name="new_saves/finetune_freeze"
bottleneck_option=norelu_C8S1
cutlayer_list="4"
num_agent_list="2"
train_gan_AE_type=conv_normN0C16
gan_loss_type=SSIM
transfer_source_task=cifar10
for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for cutlayer in $cutlayer_list; do
                        for num_agent in $num_agent_list; do
                                for local_lr in $local_lr_list; do
                                        for dataset in $dataset_list; do
                                        filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_agent}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}_servertune_${local_lr}_loadserver
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_agent=$num_agent --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                --load_from_checkpoint --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} \
                                                --load_from_checkpoint_server --transfer_source_task ${transfer_source_task}

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
                                        done
                                done
                        done
                done
        done
done