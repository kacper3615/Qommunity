# Qommunity
Qommunity is a library for community detection where you can use variety of solvers, including quantum solvers provided by QHyper
## Prerequierments
If you want to use quantum solvers, you have to add your DWave Leap token to your environment variables
#### How to add DWave token to Windows 10/11?
1. Search for **"Edit the system enviroment variables"**
2. Press **"Enviroment variables..."** button
3. At **"System variables"** press **"New..."** button and create new variable. It should have name "DWAVE_API_TOKEN" and value should be your token
4. Apply changes
5. Reset your computer

Additionally, to increase convenience, you can add the QHyper library to the environment variables (there will be no errors with import)
#### How to add library to Windows 10/11?
1. Search for **"Edit the system enviroment variables"**
2. Press **"Enviroment variables..."** button
3. At **"User variables for [Username]"** press **"New..."** button and create new variable. It should have name "PYTHONPATH" and value should be your path to QHyper library (for example: C:\Users\Kacper\Documents\QHyper)
4. Apply changes
5. Reset your computer

Now you are able to import QHyper using "import QHyper" instead of using relative/absolute path

## How to use library?
At this moment, the library is pretty simple; everything you have to know is included in the **"demo.ipynb**" file

## ATTENTION!
For the library to work properly, you need to use the QHyper version from the **kacper3615/advantage** branch, at this moment advantage is not implemented on the main branch!