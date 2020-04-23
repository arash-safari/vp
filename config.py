pathes_mnist = [
                './checkpoint/mnist/0',
            ]

model_mnist_top_0 = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
)

model_mnist_bottom_0 = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
)

hyper_params_mnist_0 = {
    'batch':16,
    'epoch':100,
    'lr':3e-4,
    'channel':256,
    'n_res_block':4,
    'n_res_channel':256,
    'n_out_res_block':0,
    'n_cond_res_block':3,
    'dropout':0.1,
    'amp':'O0',
    'sched':'',
    'ckpt':'vqvae_99.pt',
}

#=========================================================================

def get_mnist(ind):
    return [
                {
                    'model':{
                        'top':model_mnist_top_0,
                        'bottom':model_mnist_bottom_0
                    }
                    'params':hyper_params_mnist_0,
                    'path':pathes_mnist[0]
                },
            ]

