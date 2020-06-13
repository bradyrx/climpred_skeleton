from climpred import HindcastEnsemble
from climpred.tutorial import load_dataset

from climpred_skeleton.scoring import HindcastScoring


def main():
    """
    NOTE: Make demo for PM as well.
    """
    hind = load_dataset('CESM-DP-SST')
    hind['lead'].attrs['units'] = 'years'
    verif = load_dataset('FOSI-SST')
    print('\nBefore creating an object, init dimension is floats.\n')
    print(hind.init.head().to_index())

    obj = HindcastScoring(hind, verif, 'e2o', 'same_inits')
    print('\nAfter instantiating, it is now in `cftime`.\n')
    print(obj._initialized.init.head().to_index())

    print('\nHindcast output starts with `member` dimension:\n')
    print(hind.dims)

    print('\nAfter creating object, comparison is run:\n')
    print(obj._initialized.dims)

    print('\nScore can be run to automatically align and score the forecast:\n')
    print(obj.score())

    print('\nAnd it matches that from the current climpred:')
    print(obj.score())
    hindcast = HindcastEnsemble(hind)
    hindcast = hindcast.add_observations(verif, 'fosi')
    print(hindcast.verify(alignment='same_inits'))


if __name__ == '__main__':
    main()
