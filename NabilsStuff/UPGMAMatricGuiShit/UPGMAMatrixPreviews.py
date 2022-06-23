# Python program to create a table
 
from tkinter import *
import ast;
from tkinter import *  
from PIL import ImageTk, Image   

treeCount = 0;
 
class Table:
    def __init__(self, root, total_rows, total_columns):
    
        # code for creating table
        for i in range(total_rows):
            for j in range(total_columns):
 
                self.e = Entry(root, width=16, fg='blue',
                font=('Arial',10,'bold'))
 
                self.e.grid(row=i, column=j)
                self.e.insert(END, List[i][j])
 



#Takes input file from commanndline
filename = "matrixCount.o";

#Open File Input
print("--- Opening File " + filename + " ---");
f = open(filename, "r")
print("");


#Read File Input
print("--- Reading File ---");
print("");


matrixCount = 0;

#Print File Input
for line in f:
    print(line);
    line = line.replace('\n', '');
    line = line.replace('\r', '');
    matrixCount = int(line);
print("");

print("Matrix Count is: " + str(matrixCount));


listOfMatrices = [];

print(listOfMatrices);

#Takes input file from commanndline
filename = "mergeCount.o";


#Open File Input
print("--- Opening File " + filename + " ---");
f = open(filename, "r")
print("");


#Read File Input
print("--- Reading File ---");
print("");


mergeCount = 0;

#Print File Input
for line in f:
    print(line);
    line = line.replace('\n', '');
    line = line.replace('\r', '');
    mergeCount = int(line);
print("");

print("Merge Count is: " + str(mergeCount));




# Matrix Files

listOfMatrices = [];

print(listOfMatrices);

for i in range(matrixCount):

    i = i + 1;


    #Open File Input
    filename = "Matrix.o" + str(i);
    print("--- Opening File " + filename + " ---");
    f = open(filename, "r")
    print("");


    #Read File Input
    print("--- Reading File ---");
    print("");

    #Print File Input
    for line in f:
        print(line);
        line = line.replace('\n', '');
        line = line.replace('\r', '');
        line = ast.literal_eval(line);
        print(line);

        # Add this matrix from the file to the list of matrices
        listOfMatrices.append(line);

    print("");

print(listOfMatrices);


# Cluster Files

listOfClusters = ["" for i in range(mergeCount)];


print(listOfClusters);

for i in range(mergeCount):

    i = i + 1;


    #Open File Input
    filename = "Cluster.o" + str(i);
    print("--- Opening File " + filename + " ---");
    f = open(filename, "r")
    print("");


    #Read File Input
    print("--- Reading File ---");
    print("");

    #Print File Input
    for line in f:
        print(line);
        line = line.replace('\n', '');
        line = line.replace('\r', '');
        print(line);

        # Add this matrix from the file to the list of matrices
        listOfClusters[i-1] = line;

    print("");

print(listOfClusters);





 

listofTables = [];



# Individual Tables
# for i in range(matrixCount):

#     List = listOfMatrices[i];

#     total_rows = len(List);
#     total_columns = total_rows;
     
#     # create root window S
     
#     root = Tk();
     
#     #root.attributes("-fullscreen",True)
#     root.configure(bg='black');
     
#     t = Table(root, total_rows, total_columns)

#     listofTables.append(t);





# Try to print all matrices in 1 window
root = Tk();


myLabel1 = Label(root, text = "                                                                                                                               ", anchor="e", justify=LEFT);
myLabel1.grid(row=0, column=6);

# Initialize window row printing index
row = 2;


# For each Matrix
for i in range(matrixCount):
    
    # Labels row
    myLabel1 = Label(root, text = "Matrix for Iteration #" + str(i+1), anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=0);
    myLabel1 = Label(root, text = "Merge for Iteration #" + str(i+1), anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=3);
    myLabel1 = Label(root, text = "Tree for Iteration #" + str(i+1), anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=5);
    row = row + 1;
    myLabel1 = Label(root, text = "--------------------------------------------", anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=0);
    myLabel1 = Label(root, text = "--------------------------------------------", anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=3);
    myLabel1 = Label(root, text = "--------------------------------------------", anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=5);
    row = row + 1;

    # Print for Matrix at each row

    temp = row;

    for x in range(len(listOfMatrices[i])):

        # Matrix column
        myLabel1 = Label(root, text = str(listOfMatrices[i][x]), anchor="e", justify=LEFT);
        myLabel1.grid(row=row, column=0);


        row = row + 1;


    # Print for Merge and Picture Statement at each row

    nextRow = row;
    row = temp;


    myLabel1 = Label(root, text = listOfClusters[i], anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=3);
           

    # canvas = Canvas(root);  
    # canvas.grid(row=0, column=0);
    # canvas.create_image(0,0, anchor=NW, image=img)



    # Print for Tree Image

    # pictureFileName = "tree" + str(treeCount) + ".png";
    # image = Image.open(pictureFileName);
    # # Width, Height
    # img = ImageTk.PhotoImage(image)
    # myLabel1 = Label(root, image = img);
    # myLabel1.image = img;
    # myLabel1.grid(row=row, column=5);
    



    # Go back to real row
    row = nextRow;


    treeCount = treeCount + 1;

    myLabel1 = Label(root, text = "                            ", anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=0);
    row = row + 1;
    myLabel1 = Label(root, text = "                            ", anchor="e", justify=LEFT);
    myLabel1.grid(row=row, column=0);
    row = row + 1;



root.mainloop()
