# cd D:/YaoPu/GitHub/work8_SSL_3highlights/code
# bash run.sh


#####################################################################################################################################################


# PART1. supervised learning

# # 1.vnet

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/vnet/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/vnet/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/vnet/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/vnet/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/vnet/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/vnet/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/vnet/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/vnet/RCA_100'

# # 2.unet

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/unet/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/unet/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/unet/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/unet/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/unet/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/unet/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/unet/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/unet/RCA_100'

# # 3.swinunet

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/swinunet/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/swinunet/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/swinunet/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/swinunet/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/swinunet/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/swinunet/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/swinunet/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/swinunet/RCA_100'

# # 4.isunet

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv1/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv1/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv1/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv1/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv1/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv1/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv1/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv1/RCA_100'

# # 5.isunetv2

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv2/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv2/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv2/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv2/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv2/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv2/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv2/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv2/RCA_100'

# # 6.isunetv3

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv3/LCA' --num4train=20
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv3/RCA' --num4train=20

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv3/LCA_20'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv3/RCA_20'

# python train_supervised_softmax.py --root_path='../data/CA_DSA/LCA/' --exp='supervised/isunetv3/LCA' --num4train=100
# python train_supervised_softmax.py --root_path='../data/CA_DSA/RCA/' --exp='supervised/isunetv3/RCA' --num4train=100

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='supervised/isunetv3/LCA_100'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='supervised/isunetv3/RCA_100'


#####################################################################################################################################################


# PART2. single network semi-supervised (UAMT)

# # 1.vnet

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/Vnet/LCA'
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/Vnet/RCA'

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/Vnet/LCA'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/Vnet/RCA'

# # 2.unet

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/Unet/LCA'
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/Unet/RCA'

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/Unet/LCA'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/Unet/RCA'

# # 3.swinunet

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/SwinUnet/LCA'
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/SwinUnet/RCA'

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/SwinUnet/LCA'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/SwinUnet/RCA'

# # 4.isunetv1

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/isunetv1/LCA' --consistency_rampup=200 --base_lr=1e-3
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/isunetv1/RCA' --consistency_rampup=200 --base_lr=1e-3

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/isunetv1/LCA_3layers_200.0_0.001'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/isunetv1/RCA_3layers_200.0_0.001'

# # 5.isunetv2

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/isunetv2/LCA' --consistency_rampup=200 --base_lr=1e-3
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/isunetv2/RCA' --consistency_rampup=200 --base_lr=1e-3

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/isunetv2/LCA_3layers_200.0_0.001'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/isunetv2/RCA_3layers_200.0_0.001'

# # 6.isunetv3

# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/UAMT/isunetv3/LCA' --consistency_rampup=200 --base_lr=1e-3
# python train_semisupervised_UAMT.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/UAMT/isunetv3/RCA' --consistency_rampup=200 --base_lr=1e-3

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/UAMT/isunetv3/LCA_3layers_200.0_0.001'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/UAMT/isunetv3/RCA_3layers_200.0_0.001'


#####################################################################################################################################################


# PART3. multi networks semi-supervised

# # 1.vnet_swinunet

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/vnet_swinunet/LCA' --nets='vnet_swinunet' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/vnet_swinunet/RCA' --nets='vnet_swinunet' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_swinunet/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_swinunet/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_swinunet/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_swinunet/RCA' --whichmodel='model2'

# # 2.unet_swinunet

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/unet_swinunet/LCA' --nets='unet_swinunet' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/unet_swinunet/RCA' --nets='unet_swinunet' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_swinunet/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_swinunet/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_swinunet/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_swinunet/RCA' --whichmodel='model2'

# # 3.vnet_isunet

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunet/LCA' --nets='vnet_inceptionswinunet' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunet/RCA' --nets='vnet_inceptionswinunet' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunet/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunet/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunet/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunet/RCA' --whichmodel='model2'

# # 4.unet_isunet

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunet/LCA' --nets='unet_inceptionswinunet' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunet/RCA' --nets='unet_inceptionswinunet' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunet/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunet/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunet/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunet/RCA' --whichmodel='model2'

# # 5.vnet_isunetv2

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/LCA' --nets='vnet_inceptionswinunetv2' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/RCA' --nets='vnet_inceptionswinunetv2' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv2/RCA' --whichmodel='model2'

# # 6.unet_isunetv2

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/LCA' --nets='unet_inceptionswinunetv2' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/RCA' --nets='unet_inceptionswinunetv2' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv2/RCA' --whichmodel='model2'


# # 7.vnet_isunetv3

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/LCA' --nets='vnet_inceptionswinunetv3' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/RCA' --nets='vnet_inceptionswinunetv3' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/vnet_inceptionswinunetv3/RCA' --whichmodel='model2'

# # 8.unet_isunetv3

# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/LCA' --nets='unet_inceptionswinunetv3' --batch_size=2
# python train_semisupervised_CNN_Transformer.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/RCA' --nets='unet_inceptionswinunetv3' --batch_size=2

# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/LCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/LCA' --whichmodel='model2'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/RCA' --whichmodel='model1'
# python test_model.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer/unet_inceptionswinunetv3/RCA' --whichmodel='model2'

##########

# # 9.vnet4out_isunetv5, pyramidLoss

# python train_semisupervised_CNN_Transformer_PL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/LCA' --nets='vnet4out_isunetv5' --batch_size=4
# python train_semisupervised_CNN_Transformer_PL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/RCA' --nets='vnet4out_isunetv5' --batch_size=4

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PL/vnet4out_isunetv5/RCA' --whichmodel='model2'

# # 10.unet4out_isunetv5, pyramidLoss

# python train_semisupervised_CNN_Transformer_PL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/LCA' --nets='unet4out_isunetv5' --batch_size=2
# python train_semisupervised_CNN_Transformer_PL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/RCA' --nets='unet4out_isunetv5' --batch_size=2

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PL/unet4out_isunetv5/RCA' --whichmodel='model2'

# # 11.vnet4out_isunetv5, confident learning

# python train_semisupervised_CNN_Transformer_CL_1.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/LCA_1' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0'
# python train_semisupervised_CNN_Transformer_CL_1.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/RCA_1' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0'

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/LCA_1' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/LCA_1' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/RCA_1' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_CL/vnet4out_isunetv5/RCA_1' --whichmodel='model2'

# # 12.unet4out_isunetv5, confident learning

# python train_semisupervised_CNN_Transformer_CL_1.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/LCA_1' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2'
# python train_semisupervised_CNN_Transformer_CL_1.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/RCA_1' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2'

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/LCA_1' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/LCA_1' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/RCA_1' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_CL/unet4out_isunetv5/RCA_1' --whichmodel='model2'

# # 13.vnet4out_isunetv5, pyramidLoss, confident learning, labeled_num=10

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=10 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=10

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model2'

# # 13.vnet4out_isunetv5, pyramidLoss, confident learning, labeled_num=20

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=20 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=20

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model2'

# # 13.vnet4out_isunetv5, pyramidLoss, confident learning, labeled_num=30

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=30 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --nets='vnet4out_isunetv5' --batch_size=4 --gpus='0' --labeled_num=30

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/vnet4out_isunetv5/RCA' --whichmodel='model2'

# # 14.unet4out_isunetv5, pyramidLoss, confident learning, labeled_num=10

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=10 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=10 

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model2'

# # 14.unet4out_isunetv5, pyramidLoss, confident learning, labeled_num=20

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=20 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=20 

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model2'

# # 14.unet4out_isunetv5, pyramidLoss, confident learning, labeled_num=30

# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/LCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=30 
# python train_semisupervised_CNN_Transformer_PLCL.py --root_path='../data/CA_DSA/RCA/' --exp='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --nets='unet4out_isunetv5' --batch_size=2 --gpus='1,2' --labeled_num=30 

# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/LCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/LCA' --whichmodel='model2'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model1'
# python test_model_multiout.py --root_path='../data/CA_DSA/RCA/' --model='semisupervised/CNN_Transformer_PLCL/unet4out_isunetv5/RCA' --whichmodel='model2'
