import numpy as np
import wandb


class ColorWrapper:
    def __init__(self, sim):

        self.l = 0.
        self.h = 1.
        self.sim = sim

    def wrap(self, obj_batch):
        return 200 * np.ones((obj_batch.shape[0], 3))
    

class WandBRenderer:
    def __init__(self, cfg, color=None):
        self.dataset_type = cfg.dataset
        self.flip = vars(cfg).get('xz_flip', True)
        self.color = color
        self.pc_overlay = None

        if 'shapenet' == self.dataset_type :
            self.render = self.render3d
        elif 'ik' == self.dataset_type:
            self.render = self.render_img
        else:
            assert False, f'Unknown dataset "{self.dataset_type}"; cannot set rendering function'

    def set_pc_overlay(self, overlay, color):
        color_len = overlay.shape[-2]
        mesh_color = np.tile(color, (color_len, 1))
        self.pc_overlay = np.concatenate((overlay, mesh_color), axis=-1)

    def render_img(self, samples, epoch=None, title=''):
        wandb.log({'model_sample' if not title else title: [wandb.Image(smp.reshape(smp.shape[0], -1, 4), mode='RGBA') for smp in samples]}, step=epoch)

    def render3d(self, samples, epoch=None, title='', same_frame=False, label='', color=None, overlay=False):
        if isinstance(samples, np.ndarray):
            color_len = samples.shape[-2]
            if self.flip:
                samples = np.flip(samples, axis=-1)
        else:
            # This should only happen when same_frame is True. Maybe add an assertion?
            color_len = max([smp.shape[-2] for smp in samples])
            if self.flip:
                samples = [np.flip(smp, axis=-1) for smp in samples]

        if color is not None:
            assert color.shape == (len(samples), 3)
            mesh_colors = np.tile(color, (color_len, 1, 1))
        elif self.color is not None:
            if isinstance(self.color[0], list) and not same_frame:
                color = self.color[0]
            else:
                color = self.color

            if same_frame:
                assert len(color) == len(samples), "Color should either be single 3 integer list or one per sample"
                mesh_colors = np.stack([np.tile(c, (color_len, 1)) for c in self.color], axis=1)
            else:
                mesh_colors = np.tile(color, (color_len, len(samples), 1))
        else:
            mesh_colors = np.linspace([255, 0, 255], [255, 255, 0], num=color_len, axis=0, dtype=np.uint8)
            mesh_colors = np.tile(mesh_colors, (1, len(samples), 1))

        if not same_frame:
            gbox = np.array([{'corners': [[-0.6, -0.6, 0], [-0.6, 0.6, 0], [0.6, 0.6, 0], [0.6, -0.6, 0],
                                          [-0.6, -0.6, 1], [-0.6, 0.6, 1], [0.6, 0.6, 1], [0.6, -0.6, 1]],
                              'label': label, 'color': [0, 0, 255]}])
            if isinstance(label, list):
                assert len(label) == len(samples), "Can use single label or label per sample"

                def box(ind):
                    gbox[0]['label'] = str(label[ind])
                    return gbox
            else:
                def box(ind):
                    return gbox

            if len(samples.shape) < 3:
                samples = np.expand_dims(samples, 0)

            def maybe_add_overlay(x):
                return np.concatenate((x, self.pc_overlay), axis=-2) if overlay else x

            wandb.log({'model_sample' if not title else title:
                       [wandb.Object3D({'type': 'lidar/beta',
                                        'points': maybe_add_overlay(np.concatenate((smp, mesh_colors[:, ind, :]), axis=-1)),
                                        # 'boxes': box(ind)
                                       }) for ind, smp in enumerate(samples)]}, step=epoch)
        else:   # Same frame
            pcs = []
            for ind, smp in enumerate(samples):
                # Overlay objects with different colors
                pcs.append(np.concatenate((smp, mesh_colors[:smp.shape[-2], ind]), axis=-1))
            if overlay:
                pcs.append(self.pc_overlay)
            points = np.concatenate(pcs, axis=-2)
            wandb.log({'model_sample' if not title else title: wandb.Object3D(points)}, step=epoch)
