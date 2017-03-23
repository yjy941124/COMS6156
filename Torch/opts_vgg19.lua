--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './imagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', '/data/imagenet/ilsvrc2012/', 'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'nn', 'Options: cudnn | nn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        0, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         18,    'Number of total epochs to run')
    cmd:option('-epochSize',       20000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       64,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'vgg_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'alexnetowtbn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'vgg_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'vgg', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'resnet', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    -- cmd:option('-netType',     'resnet_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'googlenet_mkldnn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-netType',     'googlenet', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')

    --cmd:option('-retrain',     './model_18.t7', 'provide path to model to retrain with')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M