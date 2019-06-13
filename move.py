import os
import shutil

for i in range (1, 6):
    for j in range (1, 9):
        name = str(i) + '_' + str(j)
        if os.path.isfile(os.path.join('.', name + '.jpg')) == True:
            if os.path.isdir(str(i)) == False:
                os.mkdir(str(i))
                folder = str(i)
                shutil.move(os.path.join('.', name + '.jpg'), os.path.join('.', folder, name + '.jpg'))
                
            elif os.path.isdir(str(i)) == True:
                folder = str(i)
                shutil.move(os.path.join('.', name + '.jpg'), os.path.join('.', folder, name + '.jpg'))
                
        else:
            continue;