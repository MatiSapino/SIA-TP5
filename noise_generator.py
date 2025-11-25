import argparse
import numpy as np
from typing import List


def read_digits_txt(path: str) -> List[np.ndarray]:
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    digits = []
    for i in range(0, len(lines), 7):
        block = lines[i:i+7]
        if len(block) < 7:
            break
        matrix = np.array([[float(x) for x in row.split()] for row in block])
        digits.append(matrix)
    return digits


def write_digits_txt(digits: List[np.ndarray], path: str):
    with open(path, 'w') as f:
        for digit in digits:
            for row in digit:
                # format values with 2 decimals to keep file compact
                f.write(' '.join(f'{x:.2f}' for x in row) + '\n')


def add_gaussian_noise(digits: List[np.ndarray], stddev: float, seed: int = None) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    noisy = []
    for d in digits:
        noise = rng.normal(loc=0.0, scale=stddev, size=d.shape)
        nd = d + noise
        nd = np.clip(nd, 0.0, 1.0)
        noisy.append(nd)
    return noisy


def generate_augmented_dataset(digits: List[np.ndarray], copies: int = 1, stddev: float = 0.2, include_original: bool = True, binarize: bool = False, threshold: float = 0.5, seed: int = None, force_change: bool = False) -> List[np.ndarray]:
    out = []
    rng = np.random.RandomState(seed)
    max_tries = 25
    # Write all originals first
    if include_original:
        for d in digits:
            out.append(d.copy())
    # Then write all noisy digits (for each copy, all digits)
    for c in range(copies):
        for d in digits:
            for _ in range(max_tries):
                noise = rng.normal(loc=0.0, scale=stddev, size=d.shape)
                nd = d + noise
                nd = np.clip(nd, 0.0, 1.0)
                if binarize:
                    nd = (nd >= threshold).astype(float)
                if not force_change:
                    out.append(nd)
                    break
                if not np.array_equal(nd, d):
                    out.append(nd)
                    break
            else:
                # Si falló en todos los intentos, forzamos cambio manual
                nd = d.copy()
                idx = rng.randint(nd.size)
                nd.flat[idx] = 1.0 - nd.flat[idx]  # flip binario
                out.append(nd)
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate noisy digit dataset (Gaussian noise).')
    parser.add_argument('--input', type=str, default='data/TP3-ej3-digitos.txt')
    parser.add_argument('--output', type=str, default='data/TP3-ej3-digitos-noisy.txt')
    parser.add_argument('--stddev', type=float, default=0.2)
    parser.add_argument('--copies', type=int, default=1, help='Noisy copies to create per digit')
    parser.add_argument('--include-original', action='store_true', dest='include_original', help='Include original digits in output')
    parser.add_argument('--no-original', action='store_false', dest='include_original', help='Do not include originals')
    parser.set_defaults(include_original=True)
    parser.add_argument('--binarize', action='store_true', help='Threshold noisy pixels back to 0/1')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    digits = read_digits_txt(args.input)
    if not digits:
        print(f'No digits read from {args.input}')
        return

    augmented = generate_augmented_dataset(digits, copies=args.copies, stddev=args.stddev, include_original=args.include_original, binarize=args.binarize, threshold=args.threshold, seed=args.seed)

    write_digits_txt(augmented, args.output)
    print(f'Wrote {len(augmented)} digit blocks to {args.output} (copies={args.copies}, stddev={args.stddev}, binarize={args.binarize})')


if __name__ == '__main__':
    main()
