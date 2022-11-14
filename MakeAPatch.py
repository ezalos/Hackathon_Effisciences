import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def save_tensor(s_t, ep_i, patched=True):
    # print(f"{type(s_t) = }")
    np_s_t = (s_t.clone() * 255).int()
    np_s_t = np_s_t.cpu().numpy()
    # print(f"{type(s_t) = }")
    for c in range(4):
        print(f"{type(np_s_t) = } and {np_s_t.shape = }")
        np_s_t_c  = np_s_t[0,c] # make sure tensor is on cpu
        x = "b" if patched else "a"
        path = f"./movies/MDQN_modern/Kangaroo/0_patched/{ep_i}_{x}_{c}.png"
        # print(f"{np_s_t_c.shape = }")
        # print(f"{path = }")
        cv2.imwrite(path, np_s_t_c)

def get_mask_patch(state, x_origin, y_origin, x_size, y_size):
    mask = torch.zeros_like(state)
    patch = torch.ones(x_size, y_size)
    # print(f"{(x_origin, y_origin, x_size, y_size) = }")
    # print(f"{mask.shape = }")
    # print(f"{patch.shape = }")
    # print(f"{mask[:, :, x_origin:x_origin + x_size, y_origin:y_origin + y_size].shape = }")
    mask[:, :, x_origin:x_origin + x_size, y_origin:y_origin + y_size] = patch 
    return mask.to(torch.bool)

def get_qvals_and_argmax(agent, s_t):
    q_vals = agent.forward(s_t)
    # qvals.shape = (batch, actions)

    q_val, argmax_a = q_vals.max(1)
    # qval.shape = (max_qval_for_each_batch_elem)
    # argmax_a.shape = (idx_for_each_max_qval_batch_elem)
    a_t = argmax_a.item()

    # print(f"{q_val.shape = }")
    # print(f"{q_vals.shape = }")
    # print(f"{argmax_a.shape = }")
    return q_vals, a_t

def make_plot(st_a, st_b, at_a, at_b, idx):
    gs1 = gridspec.GridSpec(5, 4)
    # gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax_st_a = plt.subplot(gs1[0:2, 0:2])
    ax_st_b = plt.subplot(gs1[0:2, 2:4])
    ax_at_a = plt.subplot(gs1[4:5, 0:2])
    ax_at_b = plt.subplot(gs1[4:5, 2:4])

    # print(f"{st_a[0,0].shape = }")
    # print(f"{st_a[0,0].dtype = }")
    ax_st_a.imshow(st_a[0,-1], cmap='Greys',  interpolation='nearest')
    ax_st_b.imshow(st_b[0,-1], cmap='Greys',  interpolation='nearest')
    
    ax_at_a.bar([i for i in range(18)], at_a[0])
    ax_at_b.bar([i for i in range(18)], at_b[0])

    ax_st_a.set_title("Raw State")
    ax_st_b.set_title("Patched State")
    ax_at_a.set_title("Q-Values for Raw State")
    ax_at_b.set_title("Q-Values for Patched State")

    # plt.show()
    plt.savefig(f'./movies/MDQN_modern/Kangaroo/pl2/{idx:06d}.png')

def patch_loop(env, agent):
    lr = 1e-3
    steps = int(1e2)

    # freeze_all_layers_(agent)
    agent.eval()
    state, done = env.reset(), False
    # print(f"{state.shape = }")
    i = 0

    while not done:
        print(f"Episode idx: {i}")
        # print(f"{state.shape = }")
        s_t = state.clone()
        s_t = s_t.float().div(255)
        st_a = s_t.clone()

        s_t.requires_grad = True
        mask = get_mask_patch(s_t, 42, 42, 8, 8)
        # print(f"{mask.shape = }")
        # save_tensor(s_t, i, patched=False)

        opt = torch.optim.Adam([s_t], lr=lr)
        
        q_vals, a_t = get_qvals_and_argmax(agent, s_t)

        qv_a = q_vals.clone()

        for _ in tqdm(range(steps)):
            # Error

            error = q_vals[:, a_t]
            # print(f"{error = }")
            # print(f"{error.shape = }")
            # print(f"{error.dtype = }")

            opt.zero_grad()
            error.backward()
            s_t.grad[~mask] = 0
            # print(f"{s_t.grad[mask] = }")
            opt.step()

            q_vals, a_t = get_qvals_and_argmax(agent, s_t)

        # save_tensor(s_t, i, patched=True)
        make_plot(
            st_a.clone().detach().cpu().numpy(), 
            s_t.clone().detach().cpu().numpy(), 
            qv_a.clone().detach().cpu().numpy(), 
            q_vals.clone().detach().cpu().numpy(),
            i
        )

        state, reward, done, _ = env.step(a_t)
        i += 1
