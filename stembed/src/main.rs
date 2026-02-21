use stembed::{
    embedding::{StaticEmbeddingModelBuilder, WeightsDType, base_model_with_bert_uncased_tokenizer}, huggingface::hf_model_as_builder,
};

pub fn main() {
    let builder = hf_model_as_builder("sentence-transformers/static-retrieval-mrl-en-v1").unwrap();

    // save the model to a binary file
    let binary = builder.cast_weights_dtype(WeightsDType::F16).to_model_binary();
    std::fs::write("model.bin", &binary).unwrap();

    let builder2 = StaticEmbeddingModelBuilder::from_model_binary(&binary).unwrap();


    let model = base_model_with_bert_uncased_tokenizer(builder2);
    println!("{:?}", model.embed_f32("Hello, world!"));
}
