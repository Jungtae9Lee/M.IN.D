echo "Downloading cifar10 datasets..."
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "Unziopping datasets..."
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

echo "Done."
