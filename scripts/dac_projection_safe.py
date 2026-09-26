"""Same FP32 linear projection, bounded GEMM row count for CUDA compatibility."""
import torch

def project_directions(direction, weight):
    flat=direction.reshape(-1,direction.shape[-1])
    if flat.shape[0]<=6:
        output=torch.nn.functional.linear(flat,weight)
    else:
        # Avoid the failing seven-or-more-row GEMM in the model process.
        # This is an execution workaround, not a change to the DAC formula.
        output=torch.cat([torch.nn.functional.linear(x.contiguous(),weight)
                          for x in flat.split(6)],dim=0)
    return output.reshape(*direction.shape[:-1],weight.shape[0])

if __name__=='__main__':
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        torch.manual_seed(42)
        w=torch.randn(1000,128,device=device)
        for n in [1,3,6,7,8,10,12]:
            d=torch.randn(2,n,128,device=device)
            ref=torch.nn.functional.linear(d,w);got=project_directions(d,w)
            torch.testing.assert_close(got,ref,atol=1e-4,rtol=1e-4)
    print('PROJECTION_TEST_PASS',flush=True)
