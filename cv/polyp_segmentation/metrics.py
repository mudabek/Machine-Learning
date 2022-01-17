def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def jaccard(input, target):
    dice_score = dice(input, target)
    jaccard_score = dice_score / (2 - dice_score)

    return jaccard_score