import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import sys
import argparse
import re
from pylab import figure, show, legend, ylabel

from mpl_toolkits.axes_grid1 import host_subplot


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
  parser.add_argument('output_file', help='file of captured stdout and stderr')
  args = parser.parse_args()
  
  f = open(args.output_file, 'r')

  training_iterations = []
  training_loss = []

  test_iterations = []
  test_accuracy = []
  test_loss = []

  check_test = False
  check_test2 = False
  for line in f:

    if check_test and 'accuracy' in line:
      #print test_iterations, line
      test_accuracy.append(float(line.strip().split(' = ')[-1]))
      check_test = False
      check_test2 = True
    elif check_test2:
      if 'Test net output' in line and 'loss_cls' in line:
        test_loss.append(float(line.strip().split(' ')[-2]))
        check_test2 = False
      elif 'Test net output' in line and 'accuracy' in line:
        test_accuracy.append(float(line.strip().split(' ')[-1]))
        check_test2 = False
      elif 'empty' in line:
        print "empty skip line"
      else:
        test_loss.append(0)
        check_test2 = False

    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      training_loss.append(float(line.strip().split(' = ')[-1]))

    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      check_test = True



  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'test loss len: ', len(test_loss)
  print 'test iterations len: ', len(test_iterations)
  print 'test accuracy len: ', len(test_accuracy)

  print 'best performance on test set is ', max(test_accuracy)



  if len(test_iterations) != len(test_accuracy): #awaiting test...
    print 'mis-match'
    print len(test_iterations[0:-1])
    test_iterations = test_iterations[0:-1]

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.plot(test_iterations, test_accuracy, '-', linewidth=2)
#  plt.show()
 
  #print "test iterations", test_iterations
  #print "test accuracy", test_accuracy
  #print "test loss", test_loss
  #print "training loss", training_loss
 
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("loss")
  par1.set_ylabel("test accuracy")
 
  p1, = host.plot(training_iterations, training_loss, label="training log loss")
  p3, = host.plot(test_iterations, test_loss, label="test log loss")
  p2, = par1.plot(test_iterations, test_accuracy, label="test accuracy")

  host.legend( loc='lower left', bbox_to_anchor=(-0.2,0)  )
  #host.legend( loc='center left', bbox_to_anchor=(1,0.5)  )
  #host.legend( loc='right', bbox_to_anchor=(0,0)  )

  host.axis["left"].label.set_color(p1.get_color())
  par1.axis["right"].label.set_color(p2.get_color())

  curTitle = 'acc: %f (iter-%d)' %(max(test_accuracy), test_iterations[np.argmax( np.asarray(test_accuracy))] )
  plt.title(curTitle);

  plt.savefig(args.output_file+".jpg");
  plt.draw()
  plt.show()



