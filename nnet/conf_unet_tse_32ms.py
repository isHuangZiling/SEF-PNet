fs = 8000 
chunk_len = 4  # (s)
chunk_size = chunk_len * fs 

nnet_conf = {
    "win_len": 256, 
    "win_inc": 64, 
    "fft_len": 256, 
    "win_type": "sqrthann",
    "kernel_size": (3, 3),
    "stride1": (1, 1), 
    "stride2": (1, 2), 
    "paddings": (1, 0),
    "output_padding": (0, 0),
    "tcn_dims": 384, 
    "tcn_blocks": 10,
    "tcn_layers": 2,
    "causal": False,
    "num_spks": 1 
}


# data configure:
train_dir = "/node/hzl/expriment/SEF_PNet_icassp2025_github/data/train/"
dev_dir = "/node/hzl/expriment/SEF_PNet_icassp2025_github/data/dev/"

train_data = {
    "mix_scp": train_dir + "mix_clean.scp", 
    "ref_scp": train_dir + "ref.scp",
	"aux_scp": train_dir + "auxs1.scp",
    "sample_rate": fs,
}

dev_data = { 
    "mix_scp": dev_dir + "mix_clean.scp", 
    "ref_scp": dev_dir + "ref.scp",
	"aux_scp": dev_dir + "auxs1.scp",
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 0.5e-3, 
    "weight_decay": 1e-5, 
}

trainer_conf = {
    "optimizer": "adam", 
    "optimizer_kwargs": adam_kwargs, 
    "min_lr": 1e-8, 
    "patience": 2, 
    "factor": 0.5, 
    "logging_period": 200  
}
