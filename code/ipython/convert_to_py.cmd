SET PATH=%PATH%;C:/Anaconda3/Scripts/
for %%f in (dir *.ipynb) do jupyter nbconvert --to script %%f