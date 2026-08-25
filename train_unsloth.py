# train_unsloth.py

import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import os
import glob
import json
import re

try:
    import wandb
except ImportError:
    wandb = None

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="all")
parser.add_argument("--include", type=str, default="", help="The *only* subsets to include, comma separated.")
parser.add_argument("--exclude", type=str, default="", help="Subsets to exclude, comma separated")
parser.add_argument("--model_path", type=str, default="~/nobackup/models/Qwen3-30B-A3B-Instruct-2507/")
parser.add_argument("--model_type", type=str, default="qwen")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_acc", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=1)    
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--small", action="store_true", help="Train on 50k")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb", action="store_true", help="Report to wandb")
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--fft", action="store_true", help="Full finetuning")
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--lora_rank", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--stop_after_1ep", action="store_true", help="Stop training after first epoch, but keep LR schedule as if training for full --epochs")

def log_gpu_memory(stage=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            print(f"GPU {i} {stage}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Max Allocated: {max_allocated:.2f} GB")
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    f"gpu_{i}_memory_allocated_gb": allocated,
                    f"gpu_{i}_memory_reserved_gb": reserved,
                    f"gpu_{i}_memory_max_allocated_gb": max_allocated,
                })
    else:
        print(f"No GPU available {stage}")

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory in output_dir"""
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Sort by step number (extracted from directory name)
    def get_step(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1
    
    checkpoint_dirs.sort(key=get_step, reverse=True)
    return checkpoint_dirs[0]

def get_wandb_run_id_from_checkpoint(checkpoint_dir, output_dir=None):
    """Extract wandb run ID from checkpoint or output directory"""
    wandb_state_file = os.path.join(checkpoint_dir, "wandb_state.json")
    if os.path.exists(wandb_state_file):
        try:
            with open(wandb_state_file, 'r') as f:
                state = json.load(f)
                # Some trainers store wandb info at the top level
                if 'run_id' in state:
                    return state['run_id']
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None

def main():
    args = parser.parse_args()

    # Fix for Phi models: Disable Dynamo compilation before model loading
    # Phi models use LongRoPE which has a conditional check that Dynamo cannot trace
    if args.model_type in ["phi", "phiu"]:
        # Disable Dynamo compilation before Unsloth tries to compile
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        print("Warning: Disabled Dynamo compilation for Phi model due to LongRoPE compatibility")

    # Determine checkpoint path for resuming
    checkpoint_path = None
    wandb_run_id = None
    
    if args.resume:
        checkpoint_path = find_latest_checkpoint(args.output_path)
        if checkpoint_path:
            print(f"Resuming from latest checkpoint: {checkpoint_path}")
        else:
            assert False, f"Error: --resume flag used but no checkpoint found in {args.output_path}"
        
        # Try to get wandb run ID from checkpoint
        if checkpoint_path:
            wandb_run_id = get_wandb_run_id_from_checkpoint(checkpoint_path, args.output_path)
            if wandb_run_id:
                print(f"Found wandb run ID in checkpoint: {wandb_run_id}")
            else:
                assert False, f"Error: --resume flag used but no wandb run ID found in checkpoint: {checkpoint_path}"

    if args.wandb and wandb is not None:
        # get name of last directory in output path
        output_path = args.output_path
        last_dir = os.path.basename(os.path.normpath(output_path))
        
        # Resume wandb run if we have a run ID from checkpoint
        wandb_init_kwargs = {
            'project': 'hategpt',
            'config': vars(args),
            'name': last_dir,
        }
        
        if wandb_run_id:
            wandb_init_kwargs['id'] = wandb_run_id
            wandb_init_kwargs['resume'] = 'must'  # Must resume this specific run
            print(f"Resuming wandb run: {wandb_run_id}")
        elif checkpoint_path:
            # Checkpoint exists but no run ID found, allow resuming if run exists
            wandb_init_kwargs['resume'] = 'allow'
            print("Allowing wandb to resume if run exists")
        
        wandb.init(**wandb_init_kwargs)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path if checkpoint_path is None else checkpoint_path,
        max_seq_length = args.max_seq_length,   # Context length - can be longer, but uses more memory
        load_in_4bit = not args.fft,     # 4bit uses much less memory
        full_finetuning = args.fft, # We have full finetuning now!
        device_map = "auto",
        # token = "hf_...",      # use one if using gated models
    )

    # Additional fix: Patch LongRoPE if model is already loaded with compilation
    # This handles cases where Unsloth has already compiled the model
    if args.model_type in ["phi", "phiu"]:
        try:
            import torch._dynamo as dynamo
            # Suppress errors if compilation still happens
            dynamo.config.suppress_errors = True
            
            # Patch the LongRoPE function to avoid data-dependent branching
            from transformers import modeling_rope_utils
            
            def patched_longrope_frequency_update(self, position_ids, device=None):
                """Patched version that always updates frequencies to avoid Dynamo issues"""
                # Always update frequencies regardless of sequence length
                # This is safe because the update is idempotent and only affects very long sequences
                seq_len = torch.max(position_ids) + 1
                # Always perform the update (removed conditional check)
                if hasattr(self, '_update_freqs_for_longrope'):
                    self._update_freqs_for_longrope(seq_len, device)
            
            # Replace the function
            modeling_rope_utils.longrope_frequency_update = patched_longrope_frequency_update
            print("Patched LongRoPE function to avoid Dynamo compilation issues")
        except (ImportError, AttributeError) as e:
            print(f"Note: Could not patch LongRoPE function: {e}")

    if not args.fft:
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_rank,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = args.lora_alpha,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = args.seed,
        )

    # Load dataset from disk (same as train_trl.py)
    dataset = load_from_disk(args.dataset)

    # drop text column
    if "text" in dataset['train'].column_names:
        dataset['train'] = dataset['train'].remove_columns(["text"])
        dataset['test'] = dataset['test'].remove_columns(["text"])

    if "label" in dataset['train'].column_names:
        dataset['train'] = dataset['train'].remove_columns(["label"])
        dataset['test'] = dataset['test'].remove_columns(["label"])

    to_include = args.include.split(",")
    if to_include != [""]:
        dataset = dataset.filter(lambda x: x['dataset'] in to_include, num_proc=10)
    else:
        to_exclude = args.exclude.split(",")
        dataset = dataset.filter(lambda x: x['dataset'] not in to_exclude, num_proc=10)

    if args.debug:
        for split in dataset:
            dataset[split] = dataset[split].select(range(10))

    if args.small:
        dataset['train'] = dataset['train'].shuffle(seed=42).select(range(50000))
        dataset['test'] = dataset['test'].shuffle(seed=42).select(range(1000))

    print(dataset)


    # Convert messages to text format if needed (for unsloth compatibility)
    if 'text' not in dataset['train'].column_names and 'messages' in dataset['train'].column_names:
        def convert_messages_to_text(examples):
            texts = []
            messages_list = examples.get('messages', [])
            if messages_list is None:
                # Determine batch size from first available column
                for key in examples.keys():
                    if examples[key] is not None and len(examples[key]) > 0:
                        return {'text': [""] * len(examples[key])}
                return {'text': [""]}
            
            for i in range(len(messages_list)):
                if messages_list[i] is None:
                    texts.append("")
                    continue
                try:
                    if args.model_type == "qwen_old":
                        # append /no_think to user messages
                        # for message in messages_list[i]:
                        #     if message['role'] == "user":
                        #         message['content'] = message['content'] + " /no_think"
                        text = tokenizer.apply_chat_template(messages_list[i], tokenize=False, enable_thinking=False)
                        text = text.replace("<think>\n\n</think>\n\n", "")
                    else:
                        text = tokenizer.apply_chat_template(messages_list[i], tokenize=False)
                    if text is None:
                        texts.append("")
                    else:
                        texts.append(text)
                except (TypeError, AttributeError, KeyError) as e:
                    print(f"Error processing message {i}: {e}, message: {messages_list[i]}")
                    texts.append("")
            return {'text': texts}
        
        dataset = dataset.map(convert_messages_to_text, batched=True, num_proc=10)
        # Filter out empty texts
        dataset = dataset.filter(lambda x: x.get('text', '') != "", num_proc=10)

    
    class LoggingCallback(TrainerCallback):
        """Callback to log GPU memory usage and time during training"""
        def __init__(self):
            super().__init__()
                
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % args.logging_steps == 0:
                log_gpu_memory(f"(step {state.global_step})")
    
    class WandbIdCallback(TrainerCallback):
        """Callback to save wandb_id to wandb_state.json"""
        def on_save(self, args, state, control, **kwargs):
            # If W&B not initialized, nothing to save
            if wandb is None or wandb.run is None:
                print("Wandb not initialized, skipping save")
                return

            save_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            wandb_state_path = os.path.join(save_dir, "wandb_state.json")
            
            wandb_state = {
                "run_id": wandb.run.id,
                "run_name": wandb.run.name,
                "project": wandb.run.project,
                "entity": wandb.run.entity,
            }

            # Save updated trainer_state.json
            with open(wandb_state_path, "w") as f:
                json.dump(wandb_state, f, indent=2)

            return control
    
    class StopAfterFirstEpochCallback(TrainerCallback):
        """Callback to stop training after the first epoch"""
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch >= 1.0:
                print(f"Stopping training after first epoch (epoch {state.epoch:.2f})")
                control.should_training_stop = True
            return control
    
    # Prepare callbacks list
    callbacks = [LoggingCallback(), WandbIdCallback()]
    if args.stop_after_1ep:
        callbacks.append(StopAfterFirstEpochCallback())
                
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset.get('test', None), # Use test split if available
        callbacks = callbacks,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_acc, # Use GA to mimic batch size!
            warmup_ratio = 0.01,
            num_train_epochs = args.epochs, # Set this for 1 full training run.
            # max_steps = 30,
            learning_rate = args.lr, # Reduce to 2e-5 for long training runs
            logging_steps = args.logging_steps,
            eval_steps = args.eval_steps,
            save_steps = args.save_steps,
            optim = "adamw_8bit",
            save_total_limit = 2,
            # max_grad_norm = 0.5,
            weight_decay = 0.001,
            lr_scheduler_type = "cosine",
            seed = args.seed,
            output_dir = args.output_path if args.output_path else None,
            report_to = "wandb" if args.wandb else "none",
            resume_from_checkpoint = checkpoint_path if checkpoint_path else None,
        ),
    )

    if args.model_type == "qwen":
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif args.model_type == "qwen_old":
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"#<think>\n\n</think>\n\n"
    elif args.model_type == "llama":
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif args.model_type == "phi":
        instruction_part="<|im_start|>user<|im_sep|>"
        response_part="<|im_start|>assistant<|im_sep|>"
    elif args.model_type == "phiu":
        instruction_part="<|user|>"
        response_part="<|assistant|>"
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
    # print(tokenizer.decode(trainer.train_dataset[100]["labels"]))
        
    trainer = train_on_responses_only(
        trainer,
        instruction_part = instruction_part,
        response_part = response_part,
    )
    
    log_gpu_memory("(before training)")
    trainer.train(resume_from_checkpoint = checkpoint_path if checkpoint_path else None)
    log_gpu_memory("(after training)")

if __name__ == "__main__":
    main()