from multiprocessing import freeze_support
import os
import subprocess

HF_TOKEN = os.environ.get("HF_TOKEN")

# 勝手にアップデートするのを止める
os.environ['UNSLOTH_DISABLE_AUTO_UPDATES'] = '1'

from unsloth import FastLanguageModel
import torch

if __name__ == '__main__':
    freeze_support()

    # TODO: なんか動かないけど多分方法はある
    # if torch.cuda.get_device_capability()[0] >= 8:
    #     subprocess.run(["uv", "pip", "install", "--no-deps", "packaging", "ninja", "einops", "flash-attn>=2.6.3"])

    max_seq_length = 2048  # モデルが処理できるシーケンスの最大長
    dtype = None  # 適切なデータ型を自動的に検出
    load_in_4bit = True  # 4ビット量子化でメモリ効率を向上

    # モデルとトークナイザをロード
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # 使用するモデルを選択。
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...",                     # 特定の制限付きモデル（例: Meta-Llama）用のアクセストークンを指定
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # 任意の値を指定可能（0以上）。推奨値: 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],  # 継続的な事前学習を行う場合に追加
        lora_alpha=32,  # LoRAのスケールファクター
        lora_dropout=0,  # 任意の値を設定可能
        bias="none",  # 任意の値を設定可能
        # [新機能] "unsloth" を使用するとVRAM消費が30%削減され、バッチサイズを2倍に拡張可能！
        # use_gradient_checkpointing="unsloth",  # Trueまたは"unsloth"を指定して超長文のコンテキストに対応
        use_gradient_checkpointing=False,  # FIXME: 有効にすると動かない
        random_state=3407,  # 再現性を確保するための乱数シード
        use_rslora=True,  # ランク安定化LoRAをサポート
        loftq_config=None,  # LoftQもサポート
    )

    from datasets import load_dataset

    dataset = load_dataset(
        "kajuma/CC-news-2024-July-October-cleaned",
        split="train",
    )

    dataset = dataset.train_test_split(train_size=0.30)["train"]

    news_prompt = """
    ###text:{}
    ###
    """

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        charsets = examples["charset"]
        texts = examples["text"]
        outputs = []
        for charset, text in zip(charsets, texts):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = news_prompt.format(text) + EOS_TOKEN
            outputs.append(text)
        return {
            "text": outputs,
        }

    pass

    # フォーマット関数を適用してデータを整形
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    trainer = UnslothTrainer(
        model=model,  # トレーニング対象のモデル
        tokenizer=tokenizer,  # モデル用トークナイザー
        train_dataset=dataset,  # トレーニングデータセット
        dataset_text_field="text",  # データセット内のテキストフィールド名
        max_seq_length=max_seq_length,  # 最大シーケンス長
        dataset_num_proc=2,  # データセット処理に使用するプロセス数
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,  # 各デバイスごとのバッチサイズ
            gradient_accumulation_steps=8,  # 勾配の累積ステップ数

            # 長時間のトレーニングに使用可能な設定
            max_steps=120,  # トレーニングの最大ステップ数
            warmup_steps=10,  # ウォームアップステップ数
            # warmup_ratio = 0.1,                # ウォームアップ比率（オプション）
            # num_train_epochs = 1,              # トレーニングのエポック数（オプション）

            # 埋め込み行列には通常より2～10倍小さい学習率を選択
            learning_rate=5e-5,  # 全体の学習率
            embedding_learning_rate=1e-5,  # 埋め込み層の学習率
            fp16=not is_bfloat16_supported(),  # FP16を使用（bfloat16がサポートされていない場合）
            bf16=is_bfloat16_supported(),  # bfloat16を使用（サポートされている場合）
            logging_steps=1,  # ログを記録するステップ間隔
            optim="adamw_8bit",  # 8ビット版AdamWオプティマイザーを使用
            weight_decay=0.01,  # 重み減衰率
            lr_scheduler_type="linear",  # 学習率スケジューラのタイプ
            seed=3407,  # 再現性のための乱数シード
            output_dir="outputs",  # 出力ディレクトリ
            report_to="none",  # ログ出力先（例: "wandb"などを指定可能）
        ),
    )

    trainer_stats = trainer.train()
