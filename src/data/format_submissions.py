from pandas import DataFrame


def format_submissions(prediction, classes) -> DataFrame:
    """
    Formats a list of image predictions into a pandas DataFrame.

    Args:
        prediction (list): A list of dictionaries containing image IDs and predictions.
        classes (list): A list of strings representing the different classes that the predictions can belong to.

    Returns:
        DataFrame: A pandas DataFrame with the image IDs as the first column and the predictions for each class as the remaining columns.
    """
    all_image_ids = [dic["image_id"] for dic in prediction]
    all_image_ids_flat = [item for sublist in all_image_ids for item in sublist]

    all_tensors = [dic["prediction"] for dic in prediction]
    all_tensors_flat = [item for sublist in all_tensors for item in sublist]

    submission_data = {"": all_image_ids_flat}

    for i, col in enumerate(classes):
        submission_data[col] = [tensor[i].item() for tensor in all_tensors_flat]

    return DataFrame(submission_data)
