import moxing as mox

# src 起点 dst 目的地    from obs

src_url = "obs://xuziqiang-day02/deeplabv3/"
dst_url = "./deeplabv3"

mox.file.copy_parallel(src_url, dst_url)