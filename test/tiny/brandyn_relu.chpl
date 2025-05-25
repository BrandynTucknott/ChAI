use Tensor;
use Layer;

proc main() {
    writeln("Small ReLU test. ");
    var relu = new shared ReLU();
    writeln(relu(Tensor.zeros(4) - 1));
    writeln("Should be: [0, 0, 0, 0]"); // output according to pytorch
}