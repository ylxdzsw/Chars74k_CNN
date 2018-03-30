using OhMyJulia
using Images
using PyCall
using ProgressMeter

const fixed_dir = "D:/data-science-bowl-2018/fixed/stage1_train"
const test_dir = "D:/data-science-bowl-2018/raw/stage1_test"

const train_ids = readdir(fixed_dir)

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__, "data-science-bowl-2018/share_feature")
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
    while all(x->x>=32, si)
        mask_image = zeros(u8, si)
        for mask in masks
            box = get_box(mask)
            max(box[2] - box[1], box[4] - box[3]) > 16 && continue
            for (x, y) in mask @when max(x-box[1]+1, box[2]-x, y-box[3]+1, box[4]-y) <= 8
                mask_image[x, y] = 1
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

function train_alter(N=5000)
    loss = 0
    @showprogress for i in 1:N
        image, masks = get_one()
        cover, masks = random_drop(image, masks)

        levels = data_for_detection(image, masks)
        loss += model.learn_detect(image, cover, levels)

        for j in 1:min(10, length(masks) ÷ 2 + 1)
            mask = rand(masks)
            centers, r = data_for_masking(image, mask)
            centers = length(centers) > 64 ? unique(rand(centers, 64)) : centers
            mask = fuse_mask!(zeros(u8, cdr(size(image))), mask)
            loss += model.learn_mask(image, mask, cover, centers, r)
        end

        if i % 50 == 0
            println(' ', loss)
            loss = 0
        end
    end
end

function predict_one(image, th1=.3, th2=.5)
    si = size(image) |> cdr
    cover = zeros(u8, si)
    rawfeat = model.get_feature(image)

    r, preds = 8, []
    while all(x->x>=64, si)
        feat = model.get_feature(image)

        while true
            p = model.detect(feat, cover)
            v, i = findmax(p)
            sigmoid(v) < th1 && break

            x, y = ind2sub(size(p), i)
            p = model.mask(rawfeat, x, y, r, cover)
            p = map(x->ind2sub(size(p), x), find(p .> th2))
            p = transback_offset(p, x, y, r)
            length(p) == 0 && break

            fuse_mask!(cover, p)
            push!(preds, p)
        end

        si = (si .+ 1) .÷ 2
        r *= 2
        image = model.half(image)
    end

    preds
end

function transback_offset(mask, x, y, r)
    map(mask) do m
        i, j = m
        x + i - r, y + j - r
    end
end

function running_encoding(mask)

end

function submit()
    fout = open("D:/data-science-bowl-2018/result/submission.csv")
    fout << "ImageId,EncodedPixels\n"

    for id in readdir(test_dir)
        image = readdir("$test_dir/$id/images")[]
        image = load("$test_dir/$id/images/$image") |> im2n


    end
end

function check_one(id=rand(train_ids))
    println(id)
    image, masks = get_one(id)
    h, w = cdr(size(image))
    preds = predict_one(image)
    a = zeros(3, 2h, 2w)

    a[:, 1:h, 1:w] = image

    for mask in masks, (x, y) in mask
        a[:, h+x-1, y] = 1
        a[1, h+x-1, w+y-1] = .6
    end

    for pred in preds, (x, y) in pred
        a[:, x, w+y-1] = 1
        a[2, h+x-1, w+y-1] = .6
    end

    colorview(RGB, a)
end

function save_model()
    model.save("D:/data-science-bowl-2018/result/")
end

# TODO:
# 1. assign more weights to small nucleis
# 2. adjust learning rate (in trainer.step) by nuclei size when learning mask
# 3. try capsules
# 4. sigmoid(0) = 0.5
# 5. search a threashold for mask
