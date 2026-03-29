#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Gelu.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseGelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser Gelu op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   bool approximate = false;
   for (int i = 0; i < nodeproto.attribute_size(); ++i) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "approximate") {
         approximate = (nodeproto.attribute(i).s() == "tanh");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Gelu<float>(input_name, output_name, approximate));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gelu does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
