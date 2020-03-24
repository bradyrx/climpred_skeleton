from climpred.tutorial import load_dataset

from climpred_skeleton.alignment import LeadAlignment


def main():
    """
    NOTE: Make demo for PM as well.
    """
    hind = load_dataset('CESM-DP-SST')
    verif = load_dataset('FOSI-SST')
    print('\nBefore creating an object, init dimension is floats.\n')
    print(hind.init.head().to_index())

    obj = LeadAlignment(hind, verif, 'e2o', 'same_verifs')
    print('\nAfter instantiating, it is now in `cftime`.\n')
    print(obj._initialized.init.head().to_index())

    print('\nHindcast output starts with `member` dimension:\n')
    print(hind.dims)

    print('\nAfter creating object, comparison is run:\n')
    print(obj._initialized.dims)


if __name__ == '__main__':
    main()
