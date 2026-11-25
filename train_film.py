
import basicsr.train as train_module

from basicsr_film.data.paired_image_dataset_film import (
    Dataset_PairedImageFiLM
)

from basicsr_film.models.image_restoration_film_model import (
    ImageCleanFiLMModel
)


# --------------------------------------------------
# Model D dataset creation
# --------------------------------------------------
def create_film_dataset(dataset_opt):

    dataset_type = dataset_opt['type']

    if dataset_type == 'Dataset_PairedImageFiLM':
        return Dataset_PairedImageFiLM(dataset_opt)

    raise ValueError(
        f'Unsupported Model D dataset: {dataset_type}'
    )


# --------------------------------------------------
# Model D model creation
# --------------------------------------------------
def create_film_model(opt):

    model_type = opt['model_type']

    if model_type == 'ImageCleanFiLMModel':
        return ImageCleanFiLMModel(opt)

    raise ValueError(
        f'Unsupported Model D model: {model_type}'
    )


# --------------------------------------------------
# Override only the functions used by this
# training entry point.
# --------------------------------------------------
train_module.create_dataset = create_film_dataset
train_module.create_model = create_film_model


if __name__ == '__main__':
    train_module.main()
