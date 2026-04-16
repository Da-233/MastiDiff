import os
import shutil
import random
from tqdm import tqdm

class FastMixConfig:
    # 1. 源头：纯 DDPM-Var 的扩充结果（作为母版）
    # 确保这个目录下每类已经是 5000 张了
    ddpm_master_dir = "./ddpm_augmented_v1/train"
    
    # 2. 补充源：LDM 的生成图
    ldm_source_dir = "./ldm_augmented_v2/train"
    
    # 3. 实验配置：设定 alpha (LDM 占生成图的比例)
    # alpha = 0.2 意味着将 20% 的 DDPM 生成图替换为 LDM 生成图
    alpha = 0.8
    
    # 4. 输出目录 (脚本会自动创建)
    output_dir = f"./hybrid_alpha_{alpha}/train"
    
    categories = ['1', '2', '3', '4']
    seed = 42

random.seed(FastMixConfig.seed)

def run_fast_clone_and_swap():
    # --- Step 1: 物理克隆 ---
    if os.path.exists(FastMixConfig.output_dir):
        print(f"⚠️ 目录 {FastMixConfig.output_dir} 已存在，正在跳过克隆...")
    else:
        print(f"📂 正在克隆母版目录至 {FastMixConfig.output_dir}...")
        shutil.copytree(FastMixConfig.ddpm_master_dir, FastMixConfig.output_dir)

    # --- Step 2: 局部置换 ---
    for cat in FastMixConfig.categories:
        print(f"\n🔄 正在处理类别 {cat} (alpha={FastMixConfig.alpha})")
        target_cat_dir = os.path.join(FastMixConfig.output_dir, cat)
        ldm_cat_dir = os.path.join(FastMixConfig.ldm_source_dir, cat)

        # 找出当前目录下所有的生成图 (gen_ 开头)
        current_gen_imgs = [f for f in os.listdir(target_cat_dir) if f.startswith("gen_")]
        n_gen_total = len(current_gen_imgs)
        
        # 计算需要替换的数量
        n_swap = int(n_gen_total * FastMixConfig.alpha)
        print(f"   生成图总量: {n_gen_total} | 拟替换(LDM): {n_swap}")

        # 1. 随机选择并删除 DDPM 生成图
        to_delete = random.sample(current_gen_imgs, n_swap)
        for f in to_delete:
            os.remove(os.path.join(target_cat_dir, f))
        
        # 2. 从 LDM 源随机抽取等量样本迁入
        ldm_pool = [f for f in os.listdir(ldm_cat_dir) if f.startswith("gen_")]
        to_add = random.sample(ldm_pool, n_swap)
        
        for f in to_add:
            # 使用前缀防止文件名冲突
            shutil.copy2(os.path.join(ldm_cat_dir, f), 
                         os.path.join(target_cat_dir, f"h_{f}"))

        # 3. 最终总数校验
        final_count = len(os.listdir(target_cat_dir))
        print(f"   ✨ 完成！类别 {cat} 最终总数: {final_count}")

if __name__ == "__main__":
    run_fast_clone_and_swap()