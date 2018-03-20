using OhMyJulia
using Images
using ProgressMeter

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

function get_box(mask)
    minx, maxx = extrema(map(car, mask))
    miny, maxy = extrema(map(cadr, mask))
    minx, maxx, miny, maxy
end

function get_one(id=rand(train_ids))
    image = readdir("$fixed_dir/$id/images")[]
    masks = readdir("$fixed_dir/$id/masks")

    image = load("$fixed_dir/$id/images/$image") |> im2n
    masks = map(masks) do mask
        mask = load("$fixed_dir/$id/masks/$mask") |> im2n
        map(x->ind2sub(size(mask), x), find(mask))
    end

    image, masks
end

function data_for_detection(image, masks)
    si = size(image) |> cdr
    levels = []
    while all(x->x>=64, si)
        mask_image = zeros(u8, (2, si...))
        for mask in masks
            box = get_box(mask)
            maxsize = max(box[2] - box[1], box[4] - box[3])
            maxsize > 16 && continue
            v = maxsize > 8 ? 1 : 2
            for (x, y) in mask
                mask_image[v, x, y] = 1
            end
        end

        push!(levels, mask_image)
        si = (si .+ 1) .÷ 2
        masks = map(x->unique(map(x->(x.+1).÷2, x)), masks)
    end

    levels
end

function show_mask(mask, size)
    image = zeros(u8, size)
    for (x, y) in mask
        image[x, y] = 1
    end
    Gray.(image)
end

function fuck()
    p =  []
    @showprogress for i in train_ids, mask in cadr(get_one(i))
        push!(p, get_box(mask))
    end
    p
end
