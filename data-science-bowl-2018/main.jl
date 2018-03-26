using OhMyJulia
using Images
using PyCall
using ProgressMeter

const fixed_dir = "D:/data-science-bowl-2018/fixed/stage1_train"
const test_dir = "D:/data-science-bowl-2018/raw/stage1_test"

const train_ids = readdir(fixed_dir)

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__, "data-science-bowl-2018")
const model = pywrap(pyimport("model")[:Model]("gpu", "D:/data-science-bowl-2018/result/"))

sigmoid(x) = e^x / (e^x + 1)

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
            if maxsize > 8
                for (x, y) in mask @when max(x-box[1]+1, box[2]-x, y-box[3]+1, box[4]-y) <= 8
                    mask_image[1, x, y] = 1
                end
            else
                for (x, y) in mask
                    mask_image[2, x, y] = 1
                end
            end
        end

        push!(levels, mask_image)
        si = (si .+ 1) .÷ 2
        masks = map(x->unique(map(x->(x.+1).÷2, x)), masks)
    end

    levels
end

function data_for_masking(image, mask)
    box = get_box(mask)

    centers, r = [], 4
    while isempty(centers)
        r *= 2

        for (x, y) in mask @when max(x-box[1]+1, box[2]-x, y-box[3]+1, box[4]-y) <= r
            push!(centers, (x, y))
        end
    end

    centers, r
end

function random_drop(image, masks)
    keep = fill(true, length(masks))
    keep[rand(1:length(masks), rand(0:length(masks)-1))] = false
    cover = zeros(u8, cdr(size(image)))
    for mask in masks[.!keep]
        fuse_mask!(cover, mask)
    end
    cover, masks[keep]
end

function fuse_mask!(image, mask)
    for (x, y) in mask
        image[x, y] = 1
    end

    image
end

function train_detector(N=5000)
    loss = 0
    @showprogress for i in 1:N
        image, masks = get_one()
        cover, masks = random_drop(image, masks)
        levels = data_for_detection(image, masks)
        loss += model.learn_detect(image, cover, levels)

        if i % 100 == 0
            println(' ', loss)
            loss = 0
        end
    end
end

function train_masker(N=5000)
    loss = 0
    @showprogress for i in 1:N
        image, masks = get_one()
        for j in 1:min(10, length(masks) ÷ 2 + 1)
            cover, ms = random_drop(image, masks)
            mask = rand(ms)
            centers, r = data_for_masking(image, mask)
            centers = length(centers) > 256 ? unique(rand(centers, 256)) : centers
            mask = fuse_mask!(zeros(u8, cdr(size(image))), mask)
            loss += model.learn_mask(image, mask, cover, centers, r)
        end

        if i % 20 == 0
            println(' ', loss)
            loss = 0
        end
    end
end

function predict()
    fout = open("D:/data-science-bowl-2018/result/submission.csv")
    fout << "ImageId,EncodedPixels\n"

    for id in readdir(test_dir)
        image = readdir("$test_dir/$id/images")[]
        image = load("$test_dir/$id/images/$image") |> im2n

        si = size(image) |> cdr
        cover = zeros(u8, si)

        levels = []
        while all(x->x>=64, si)
            mask_image = zeros(u8, (2, si...))
            for mask in masks
                box = get_box(mask)
                maxsize = max(box[2] - box[1], box[4] - box[3])
                maxsize > 16 && continue
                if maxsize > 8
                    for (x, y) in mask @when max(x-box[1]+1, box[2]-x, y-box[3]+1, box[4]-y) <= 8
                        mask_image[1, x, y] = 1
                    end
                else
                    for (x, y) in mask
                        mask_image[2, x, y] = 1
                    end
                end
            end

            push!(levels, mask_image)
            si = (si .+ 1) .÷ 2
            masks = map(x->unique(map(x->(x.+1).÷2, x)), masks)
        end

        levels

        while true
            p = model.detect(image)
            v, i = findmax(p)

            sigmoid(v) < .4 && break
            h, x, y = ind2sub(size(p), i)

            if h == 1

            else

            end
        end
    end
end

function save_model()
    model.save("D:/data-science-bowl-2018/result/")
end

# TODO:
# 1. assign more weights to small nucleis
# 2. adjust learning rate (in trainer.step) by nuclei size when learning mask
# 3. try capsules
