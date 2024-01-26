def Model_factory(backbone, num_classes):
    if backbone == 'hourglass52':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 1)
        
    elif backbone == 'hourglass104':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'gaussnet':
        from basenet.gaussnet import StackedHourGlass as Model
        model = Model(num_classes, 2)
        
    elif backbone == 'gaussnet_cascade_2layers':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 1)
        
    elif backbone == 'gaussnet_cascade':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 2)
    
    elif backbone == 'gaussnet_cascade_4layers':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, 3)
        
    else:
        raise "Model import Error !! "
        
    return model



