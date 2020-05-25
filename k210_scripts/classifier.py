
import sensor, image, time, lcd
import KPU as kpu
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
# sensor.set_vflip(0)
# sensor.set_hmirror(0)
sensor.run(1)
lcd.clear()
lcd.draw_string(100,96,"MobileNet Demo")
lcd.draw_string(100,112,"Loading labels...")
f=open('labels.txt','r')
labels=f.readlines()
f.close()

task = kpu.load("/sd/Classifier_best_val_accuracy.kmodel") 
clock = time.clock()
while(True):
    img = sensor.snapshot()
    clock.tick()
    fmap = kpu.forward(task, img)
    fps=clock.fps()
    plist=fmap[:]
    pmax=max(plist)	
    max_index=plist.index(pmax)	
    a = lcd.display(img, oft=(50,0))
    lcd.draw_string(0, 224, "%.2f:%s                            "%(pmax, labels[max_index].strip()))
    print(pmax)
a = kpu.deinit(task)