using OhMyJulia
using JLD2
using Images

const fixed_dir = "D:/data-science-bowl-2018/fixed/stage1_train"
const test_dir = "D:/data-science-bowl-2018/raw/stage1_test"

const train_ids = readdir(fixed_dir)

function im2n(image)
    if ndims(image) == 3
        image = image[:, :, 1]
    end

    view = channelview(image)
    if ndims(view) == 3
        view = view[1:3, :, :]
    end

    f32.(view)
end

function get_one()
    id = rand(train_ids)
    image = readdir("$fixed_dir/$id/images")[]
    masks = readdir("$fixed_dir/$id/masks")

    image = load("$fixed_dir/$id/images/$image") |> im2n
    masks = map(masks) do mask
        mask = load("$fixed_dir/$id/masks/$mask") |> im2n
        map(x->ind2sub(size(mask), x), find(mask))
    end
end
