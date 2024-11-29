GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=intentqa \
	--dataset=intentqa \
	--mc=5 \
	--bnum=10 \
	--epochs=30 \
	--lr=0.00002 \
	--qmax_words=0 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=16 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--margin=2.4\
	--p=2\
	--dropout=0.3 \
	--seed=666 \
	--save_dir='../data/save_models/intentqa/1129test/' \
	--CM_PT=1 \
	--pretrain_path='../data/save_models/webvid180K/e1.pth'\

