#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=7
arch=vgg11_bn
batch_size=128
num_client=2
num_epochs=200
dataset_list="cifar100"
scheme=V2_epoch
random_seed_list="125"
regularization=gan_adv_step1
learning_rate=0.05
local_lr=-1
regularization_strength_list="0.3"
cutlayer_list="4"
num_client_list="2"
ssim_threshold=0.5
train_gan_AE_type=conv_normN0C16
bc_list="8"
gan_loss_type=SSIM
folder_name="new_saves/early_epoch"



# for bc in $bc_list; do
#         bottleneck_option=None
#         for dataset in $dataset_list; do
#                 for random_seed in $random_seed_list; do
#                         for regularization_strength in $regularization_strength_list; do
#                                 for cutlayer in $cutlayer_list; do
#                                         for num_client in $num_client_list; do
#                                                 filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}_ssim_${ssim_threshold}_advtrain
#                                                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
#                                                         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs --save_more_checkpoints\
#                                                         --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
#                                                         --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
#                                                         --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold}

#                                                 target_client=0
#                                                 attack_scheme=MIA
#                                                 attack_epochs=50
#                                                 average_time=1


#                                                 attack_from_later_layer_list="-1"
#                                                 num_epochs_list="0 1 2 5 10 20 50 100 200"
#                                                 for attack_from_later_layer in ${attack_from_later_layer_list}; do
#                                                         for num_epochs in $num_epochs_list; do
                                                                
#                                                                 internal_C=64
#                                                                 N=4
#                                                                 test_gan_AE_type=res_normN${N}C${internal_C}

#                                                                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
#                                                                         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
#                                                                         --dataset=$dataset --scheme=$scheme --target_client=${target_client} --attack_from_later_layer $attack_from_later_layer\
#                                                                         --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
#                                                                         --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
#                                                         done
#                                                 done
#                                         done
#                                 done
#                         done
#                 done
#         done
# done




# for bc in $bc_list; do
#         bottleneck_option=noRELU_C${bc}S1
#         for dataset in $dataset_list; do
#                 for random_seed in $random_seed_list; do
#                         for regularization_strength in $regularization_strength_list; do
#                                 for cutlayer in $cutlayer_list; do
#                                         for num_client in $num_client_list; do
#                                                 filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}_ssim_${ssim_threshold}_advtrain
#                                                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
#                                                         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs --save_more_checkpoints\
#                                                         --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
#                                                         --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
#                                                         --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold}

#                                                 target_client=0
#                                                 attack_scheme=MIA
#                                                 attack_epochs=50
#                                                 average_time=1


#                                                 attack_from_later_layer_list="-1"
#                                                 num_epochs_list="0 1 2 5 10 20 50 100 200"
#                                                 for attack_from_later_layer in ${attack_from_later_layer_list}; do
#                                                         for num_epochs in $num_epochs_list; do
                                                                
#                                                                 internal_C=64
#                                                                 N=4
#                                                                 test_gan_AE_type=res_normN${N}C${internal_C}

#                                                                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
#                                                                         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
#                                                                         --dataset=$dataset --scheme=$scheme --target_client=${target_client} --attack_from_later_layer $attack_from_later_layer\
#                                                                         --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
#                                                                         --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
#                                                         done
#                                                 done
#                                         done
#                                 done
#                         done
#                 done
#         done
# done

local_lr_list="0.005"
bottleneck_option=norelu_C8S1
interval=5
transfer_source_task=cifar10
for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for cutlayer in $cutlayer_list; do
                        for num_client in $num_client_list; do
                                for local_lr in $local_lr_list; do
                                        for dataset in $dataset_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}_servertune_${local_lr}_loadserver_source_${transfer_source_task}
                                                # CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                #         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs --save_more_checkpoints\
                                                #         --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                                #         --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                                #         --load_from_checkpoint --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name} --ssim_threshold ${ssim_threshold} \
                                                #         --load_from_checkpoint_server --transfer_source_task ${transfer_source_task} --optimize_computation ${interval}

                                                target_client=0
                                                attack_scheme=MIA
                                                attack_epochs=50
                                                average_time=1
                                                
                                                # attack_from_later_layer_list="-1"
                                                # num_epochs_list="0 1 2 5 10 20 50 100 200"
                                                # for attack_from_later_layer in ${attack_from_later_layer_list}; do
                                                #         for num_epochs in $num_epochs_list; do
                                                                
                                                #                 internal_C=64
                                                #                 N=4
                                                #                 test_gan_AE_type=res_normN${N}C${internal_C}

                                                #                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                #                         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                #                         --dataset=$dataset --scheme=$scheme --target_client=${target_client} --attack_from_later_layer $attack_from_later_layer\
                                                #                         --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                #                         --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                #         done
                                                # done
                                                
                                                internal_C=16
                                                N=0
                                                test_gan_AE_type=conv_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_client=${target_client} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}


                                                # internal_C=32
                                                # N=2
                                                # test_gan_AE_type=res_normN${N}C${internal_C}

                                                # CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                #         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs --test_best\
                                                #         --dataset=$dataset --scheme=$scheme --target_client=${target_client}\
                                                #         --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                #         --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}


                                                # internal_C=64
                                                # N=4
                                                # test_gan_AE_type=res_normN${N}C${internal_C}

                                                # CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                #         --filename=$filename --num_client=$num_client --num_epochs=$num_epochs --test_best\
                                                #         --dataset=$dataset --scheme=$scheme --target_client=${target_client}\
                                                #         --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                                #         --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}

                                        done
                                done
                        done
                done
        done
done


