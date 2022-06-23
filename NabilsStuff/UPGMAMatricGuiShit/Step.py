# Python program to create a table
 
from tkinter import *
import ast;
from tkinter import *  
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image   

treeCount = 0;

iteration = -1;

def initialStep():

    clearEverything();

    # Open the Input File Name:
    print("--- Opening Input File Name File ---");
    filename = "inputFileName.o";

    f = open(filename, "r")
    print("");


    #Read File Input
    print("--- Reading File ---");
    print("");


    #Print File Input
    actualInputFile = "";
    for line in f:
        print(line);
        line = line.replace('\n', '');
        line = line.replace('\r', '');
        actualInputFile = str(line);
    print("");



    f = open(actualInputFile, "r")
    print("");


    rawSequences = [];
    for line in f:
        print(line);
        line = line.replace('\n', '');
        line = line.replace('\r', '');
        rawSequences.append(line);
    print("");


    for string in rawSequences:
        print(string);
    print("");



    # Print all sequencnes to TreeFrame:
    for x in range(len(rawSequences)):
        myLabel1 = Label(treeview_frame, text = rawSequences[x], anchor="e", justify=LEFT);
        myLabel1.grid(row=x, column=0);
        myLabel1.config(font=("Courier", 12));


    # Print empty Matrix:
    printMatrix(listOfMatrices[0]);


    # Print example calculation:
    myLabel1 = Label(explain_frame, text = "These are all our sequences from the input file. We have " + str(len(listOfMatrices[0])-1) + " sequences.", anchor="e", justify=LEFT);
    myLabel1.grid(row=0, column=0);








def clearEverything():

    # Clear Explanation Frame
    for widget in explain_frame.winfo_children():
        widget.destroy();


    # Clear Tree Frame
    for widget in treeview_frame.winfo_children():
        widget.destroy();

    # Clear Matrix Frame
    for widget in matrix_frame.winfo_children():
        widget.destroy();

    return 1;


def printMatrix(Matrix):

    global iteration;

    matrixTitle = "Matrix #" + str(iteration);

    # Example Matrix
    myLabel1 = Label(matrix_frame, text = matrixTitle, anchor="e", justify=LEFT);
    myLabel1.grid(row=0, column=0);

    for widget in matrix_frame.winfo_children():
        widget.destroy();

    # For each Line in the Matrix
    row = 1;
    width = len(Matrix);
    for x in Matrix:

        # Print each value in its own cell
        cell = 0;

        # myLabel1 = Label(matrix_frame, text = "|", anchor="e", justify=LEFT);
        # myLabel1.grid(row=row, column=cell);
        # cell = cell + 1;

        for char in x:

            myLabel1 = Label(matrix_frame, text = str(char), anchor="e", justify=LEFT, borderwidth=2, relief="groove");
            myLabel1.grid(row=row, column=cell, sticky='ew');
            myLabel1.config(font=("Courier", 12));

            cell = cell + 1;

            # myLabel1 = Label(matrix_frame, text = "|", anchor="e", justify=LEFT);
            # myLabel1.grid(row=row, column=cell);
            # cell = cell + 1;

        row = row + 1;


    return 1;



def printTree():

    treeTitle = "Tree #" + str(iteration);
    myLabel1 = Label(treeview_frame, text = treeTitle, anchor="e", justify=LEFT);
    myLabel1.grid(row=0, column=0);
    myLabel1.config(font=("Courier", 12));


    canvas_for_image = Canvas(treeview_frame, bg='yellow', height=300, width=400, borderwidth=0, highlightthickness=0)
    canvas_for_image.grid(row=1, column=0, sticky='nesw', padx=0, pady=0)

    # create image from image location resize it
    picName = "tree" + str(iteration) + ".png";
    image = Image.open(picName);
    canvas_for_image.image = ImageTk.PhotoImage(image.resize((400, 300), Image.ANTIALIAS))
    canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')



def nextStep():

    clearEverything();

    global iteration;

    iteration = iteration + 1;

    # If iteration greater than -1
    if(iteration > -1):

        # Add previous button
        previousStep.grid(column=0, row=0, sticky="nsew");


    # If iteration at max
    if(iteration >= matrixCount):

        # remove next button
        for widget in button_frame.winfo_children():
            widget.grid_remove();

        # Add previous button
        previousStep.grid(column=0, row=0, sticky="nsew");

        # Print only the tree
        printTree();


    else:

        # Example Tree
        # create a canvas to show image on
        printTree();


        # Example Matrix
        printMatrix(listOfMatrices[iteration]);

        # Example Explanation
        myLabel1 = Label(explain_frame, text = "We Merge: " + str(listOfClusters[iteration]), anchor="e", justify=LEFT);
        myLabel1.grid(row=0, column=0);




def prevStep():

    clearEverything()
    
    global iteration;

    iteration = iteration - 1;

    # If iteration less than matrixCount
    if(iteration < matrixCount):

        for widget in button_frame.winfo_children():
            widget.grid_remove();

        previousStep.grid(column=0, row=0, sticky="nsew");
        nextStep.grid(column=1, row=0, sticky="nsew");


    # If iteration less than 0
    if(iteration < 0):

        # remove previous button
        for widget in button_frame.winfo_children():
            widget.grid_remove();

        nextStep.grid(column=1, row=0, sticky="nsew");


        # Only print initial step
        initialStep();

    else: 

        # Example Tree
        # create a canvas to show image on
        printTree();


        # Example Matrix
        printMatrix(listOfMatrices[iteration]);


        # Example Explanation
        myLabel1 = Label(explain_frame, text = "We Merge: " + str(listOfClusters[iteration]), anchor="e", justify=LEFT);
        myLabel1.grid(row=0, column=0);






#Takes input file for matrix count
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
    filename = "displayMatrix.o" + str(i);
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





# Screen Explanation
root = tk.Tk();

# Screen Size
root.geometry("900x600");

# Initialize Screen Frames
treeview_frame = tk.Frame(root, background="#FFF0C1", bd=1, relief="sunken");
matrix_frame = tk.Frame(root, background="#D2E2FB", bd=1, relief="sunken");
explain_frame = tk.Frame(root, background="#CCE4CA", bd=1, relief="sunken");
button_frame = tk.Frame(root, background="#F5C2C1", bd=1, relief="sunken");


# Place tree and matrix Frames
treeview_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
matrix_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

root.grid_rowconfigure(0, weight=3)
root.grid_rowconfigure(1, weight=3)


# Place button and explain Frames
explain_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=2, pady=2)
button_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=2, pady=2)


root.grid_columnconfigure(0, weight=8)
root.grid_columnconfigure(1, weight=10)
    

# Initialize buttons
nextStep = ttk.Button(button_frame, text="Next ->", command=nextStep);
previousStep = ttk.Button(button_frame, text="<- Previous", command=prevStep);


# Place Buttons
nextStep.grid(column=1, row=0, sticky="nsew");
# previousStep.grid(column=0, row=0, sticky="nsew");











# In the initial Screen output after putting in file input and screen input, 
# should print all sequences to Tree Frame first
# should print an empty Matrix to Matrix Frame
# should print an example calculation
initialStep();







# # Example Tree
# # create a canvas to show image on
# myLabel1 = Label(treeview_frame, text = "Tree #0", anchor="e", justify=LEFT);
# myLabel1.grid(row=0, column=0);
# myLabel1.config(font=("Courier", 12));


# canvas_for_image = Canvas(treeview_frame, bg='yellow', height=300, width=400, borderwidth=0, highlightthickness=0)
# canvas_for_image.grid(row=1, column=0, sticky='nesw', padx=0, pady=0)

# # create image from image location resize it
# image = Image.open('tree0.png')
# canvas_for_image.image = ImageTk.PhotoImage(image.resize((400, 300), Image.ANTIALIAS))
# canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')



# # Example Matrix
# printMatrix(listOfMatrices[0]);

# # Example Explanation
# myLabel1 = Label(explain_frame, text = "We Merge: " + str(listOfClusters[0]), anchor="e", justify=LEFT);
# myLabel1.grid(row=0, column=0);



root.mainloop()
