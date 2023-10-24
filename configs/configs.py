import platform
import os

# Root folder for project code vs computer host
roots = {'DESKTOP-JV9DACD':
            {'cfz-sam': 
             {
                 'Windows':'C:/repos/cfz-sam', 
                 'Linux': '/home/hailo/cfz-sam'}
                 },
         "PC-514445":
            {'cfz-sam': 
             {
                 'Windows':'C:/repos/cfz-sam', 
                 'Linux': '/home/hailo/cfz-sam'}
                 },
         'SURFACE':
             {'cfz-sam': 
              {
                  'Windows': 'C:/repos/cfz-sam',
                  'Linux': '/home/hailo/cfz-sam'}
                  }
         }

PROJECT_PATH = roots[platform.node()]['cfz-sam'][platform.system()]
IMAGES_PATH = os.path.join(PROJECT_PATH, 'images')
FASTSAM_CHECKPOINTS = os.path.join(PROJECT_PATH, 'FastSAM/weights')

print(PROJECT_PATH)