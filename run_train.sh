tp=0.0
pr=0.00005
tl=1.0
for benchmark in IC03_860 IC13_857 IC15_1811 SVT IIIT5k_3000 CUTE80
do
	CUDA_VISIBLE_DEVICES=0 python train_smile_cbsp.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
	--src_train_data ./../../../DA/Seq2SeqAdapt/data/data_lmdb_release/training \
	--tar_train_data ./../../../DA/Seq2SeqAdapt/data/data_lmdb_release/evaluation \
	--tar_select_data ${benchmark} \
	--tar_batch_ratio 1 \
	--valid_data ./../../../DA/Seq2SeqAdapt/data/data_lmdb_release/evaluation/${benchmark} \
	--continue_model ./../../../DA/Seq2SeqAdapt/saved_models/AprilYapingZhang/TPS-ResNet-BiLSTM-Attn.pth \
	--batch_size 128 --lr 1 \
	--workers 4 \
	--num_iter 30000 \
	--experiment_name /smile_cbsp_${benchmark}_init_${tp}_add_${pr}_lambda_${tl} --init_portion ${tp} --add_portion ${pr} --tar_lambda ${tl}
done
