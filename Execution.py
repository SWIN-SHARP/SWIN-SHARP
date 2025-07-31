from swintransformer import SwinTransformer
modelSwin = SwinTransformerV2(img_size=256,
                                  patch_size=4,
                                  in_chans=3,
                                  num_classes=5,
                                  embed_dim=[], # Configuration of your desired  variant of model
                                  depths=[],  # Configuration of your desired  variant of model
                                  num_heads=[],  # Configuration of your desired  variant of model
                                  window_size=16,
                                  mlp_ratio=4.,
                                  qkv_bias=True,
                                  drop_rate=0.0,
                                  drop_path_rate=0.3,
                                  ape=False,
                                  patch_norm=True,
                                  use_checkpoint=False,
                                  pretrained_window_sizes=[0,0,0,0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelSwin.to(device)
