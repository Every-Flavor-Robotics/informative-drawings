import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from model import Generator, GlobalGenerator2, InceptionV3
from dataset import UnpairedDepthDataset
from utils import channel2width


class InformativeDrawingsModel:
    def __init__(self, opt):
        if torch.cuda.is_available():
            self.device = f"cuda:{opt.cuda_device}"
        else:
            self.device = "cpu"

        self.opt = opt

        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

        self.net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
        self.net_G.load_state_dict(
            torch.load(
                os.path.join(checkpoints_dir, opt.name, "netG_A_%s.pth" % opt.which_epoch),
                map_location=self.device,
            )
        )
        self.net_G.to(self.device)
        self.net_G.eval()
        print("Loaded", os.path.join(checkpoints_dir, opt.name, "netG_A_%s.pth" % opt.which_epoch))

        self.net_GB = None
        if opt.reconstruct == 1:
            self.net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
            self.net_GB.load_state_dict(
                torch.load(
                    os.path.join(opt.checkpoints_dir, opt.name, "netG_B_%s.pth" % opt.which_epoch),
                    map_location=self.device,
                )
            )
            self.net_GB.to(self.device)
            self.net_GB.eval()

        self.netGeom = None
        self.net_recog = None
        if opt.predict_depth == 1:
            usename = opt.name
            if (len(opt.geom_name) > 0) and (
                os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))
            ):
                usename = opt.geom_name
            myname = os.path.join(opt.checkpoints_dir, usename, "netGeom_%s.pth" % opt.which_epoch)
            self.netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)
            self.netGeom.load_state_dict(torch.load(myname, map_location=self.device))
            self.netGeom.to(self.device)
            self.netGeom.eval()

            self.net_recog = InceptionV3(
                opt.num_classes,
                False,
                use_aux=True,
                pretrain=True,
                freeze=True,
                every_feat=opt.every_feat == 1,
            )
            self.net_recog.to(self.device)
            self.net_recog.eval()

    def infer(self, input_path: str, output_dir: str, batch_size: int) -> None:
        """Run inference on input_path (file or directory) and write results to output_dir."""
        opt = self.opt

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        transforms_r = [
            transforms.Resize(int(opt.size), Image.BICUBIC),
            transforms.ToTensor(),
        ]

        test_data = UnpairedDepthDataset(
            input_path,
            "",
            opt,
            transforms_r=transforms_r,
            mode=opt.mode,
            midas=opt.midas > 0,
            depthroot=opt.depthroot,
        )

        dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                if i > opt.how_many:
                    break

                img_r = batch["r"].to(self.device)
                name = batch["name"][0]

                image = self.net_G(img_r)
                save_image(image.data, os.path.join(output_dir, "%s.png" % name))

                if opt.predict_depth == 1:
                    geom_input = image
                    if geom_input.size()[1] == 1:
                        geom_input = geom_input.repeat(1, 3, 1, 1)
                    _, geom_input = self.net_recog(geom_input)
                    geom = self.netGeom(geom_input)
                    geom = (geom + 1) / 2.0
                    input_img_fake = channel2width(geom)
                    save_image(input_img_fake.data, os.path.join(output_dir, "%s_geom.png" % name))

                if opt.reconstruct == 1:
                    rec = self.net_GB(image)
                    save_image(rec.data, os.path.join(output_dir, "%s_rec.png" % name))

                if opt.save_input == 1:
                    save_image(img_r, os.path.join(output_dir, "%s_input.png" % name))

                sys.stdout.write("\rGenerated images %04d of %04d" % (i, opt.how_many))

        sys.stdout.write("\n")
