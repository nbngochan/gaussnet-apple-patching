def Model_factory(backbone, num_classes):
    if backbone == 'hourglass52':
        from basenet.hourglass import StackedHourGlass as Model
        model = Model(num_classes, 1)
    
    if backbone == 'hourglass52_cascade':
        from basenet.gaussnet_cascade import StackedHourGlass as Model
        model = Model(num_classes, num_stacks=1, cascade=2)
        
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
    
    elif backbone == 'hhrnet32':
        from basenet.higher_HRNet import Higher_HRNet32
        model = Higher_HRNet32(num_classes)
        
    elif backbone == 'hhrnet48':
        from basenet.higher_HRNet import Higher_HRNet48
        model = Higher_HRNet48(num_classes)
        
        
    else:
        raise "Model import Error !! "
        
    return model



