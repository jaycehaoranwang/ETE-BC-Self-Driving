import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import Voxelization

class SimplePillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                      point_cloud_range=point_cloud_range,
                                      max_num_points=max_num_points,
                                      max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], where each tensor is (Ni, 3) and Ni can vary
        return: 
               pillars: (P, max_num_points, 3), where P is total pillars across batch
               coors_batch: (P, 1 + 2), where each row is (batch_idx, x_idx, y_idx)
               num_points_per_pillar: (P,)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        
        for batch_idx, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_out = self.voxel_layer(pts)
            
            pillars.append(voxels_out)
            batch_coors = F.pad(coors_out, (1, 0), value=batch_idx)
            coors.append(batch_coors)
            npoints_per_pillar.append(num_points_out)
        
        pillars = torch.cat(pillars, dim=0)
        coors_batch = torch.cat(coors, dim=0)
        num_points_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        
        return pillars, coors_batch, num_points_per_pillar

class SimplePillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel=9, out_channel=64):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (P, max_num_points, 3)
        coors_batch: (P, 1 + 2)
        npoints_per_pillar: (P,)
        return: (batch_size, out_channel, y_l, x_l)
        '''
        device = pillars.device
        batch_size = coors_batch[:, 0].max().item() + 1
        
        # Calculate offsets to points center
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        # Calculate offsets to pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        # Combine features
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        
        # Create mask for valid points
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        # Feature embedding
        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooled_features = torch.max(features, dim=-1)[0]
        
        # Create batch of feature grids
        batch_canvas = []
        for batch_idx in range(batch_size):
            batch_mask = coors_batch[:, 0] == batch_idx
            curr_features = pooled_features[batch_mask]
            curr_coors = coors_batch[batch_mask]
            
            canvas = torch.zeros((self.out_channel, self.y_l, self.x_l), 
                               dtype=torch.float32, device=device)
            canvas[:, curr_coors[:, 2], curr_coors[:, 1]] = curr_features.T
            batch_canvas.append(canvas)
            
        return torch.stack(batch_canvas, dim=0)
    
class SimpleBackbone(nn.Module):
    def __init__(self, in_channel, out_channels=[64, 128, 256], layer_nums=[3, 5, 5], layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            # First conv in block with stride
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            # Additional layers in block without stride
            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

    def forward(self, x):
        '''
        x: (bs, 64, H, W)
        return: list of tensors at different scales
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs

class SimpleNeck(nn.Module):
    def __init__(self, in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128]):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                  out_channels[i], 
                                                  upsample_strides[i], 
                                                  stride=upsample_strides[i],
                                                  bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))
            self.decoder_blocks.append(nn.Sequential(*decoder_block))

    def forward(self, x):
        '''
        x: list of tensors from backbone
        return: single tensor with concatenated features
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            outs.append(self.decoder_blocks[i](x[i]))
        out = torch.cat(outs, dim=1)
        return out

class SimplePointPillars(nn.Module):
    def __init__(self,
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=16000,
                 output_dim=256):
        super().__init__()
        
        self.pillar_layer = SimplePillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )
        
        self.pillar_encoder = SimplePillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64
        )
        
        # More faithful to original architecture
        self.backbone = SimpleBackbone(
            in_channel=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2]
        )
        
        self.neck = SimpleNeck(
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128]
        )
        
        # Final layers for feature extraction
        self.final_layers = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim)
        )

    def forward(self, batched_points):
        '''
        batched_points: list[tensor], where each tensor is (Ni, 3)
        return: (batch_size, output_dim) tensor of feature vectors
        '''
        # Convert points to pillars
        pillars, coors_batch, num_points = self.pillar_layer(batched_points)
        
        # Encode pillars to get spatial feature grid
        spatial_features = self.pillar_encoder(pillars, coors_batch, num_points)
        
        # Multi-scale backbone features
        backbone_features = self.backbone(spatial_features)
        
        # Upsample and concatenate features
        neck_features = self.neck(backbone_features)
        
        # Final processing
        latent = self.final_layers(neck_features)
        
        return latent