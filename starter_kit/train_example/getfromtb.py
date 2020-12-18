from tensorboard.backend.event_processing import event_accumulator


def gettd(filepath, label):
    ea=event_accumulator.EventAccumulator(filename) 
    ea.Reload()
    # print(ea.scalars.Keys())
    
    data=ea.scalars.Items(label)
    print(len(data))
    return [(i.step,i.value) for i in data])

    