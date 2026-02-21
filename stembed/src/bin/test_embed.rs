use std::env;

use stembed::embedding::{base_model_with_bert_uncased_tokenizer, WeightsDType};
use stembed::huggingface::hf_model_as_builder;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: test_embed <model_id> <text>");
        std::process::exit(1);
    }

    let model_id = &args[1];
    let text = &args[2];

    eprintln!("Loading model: {}", model_id);
    let builder = hf_model_as_builder(model_id).unwrap();
    eprintln!(
        "  dim={}, dtype={:?}, vocab_size={}",
        builder.dim,
        builder.weights_dtype,
        builder.vocab.vocab.len()
    );

    // Always use F32 weights for maximum precision in comparison
    let builder = builder.cast_weights_dtype(WeightsDType::F32);
    let model = base_model_with_bert_uncased_tokenizer(builder);

    let tokens = model.tokenize(text);
    let embedding = model.embed_f32(text);

    // Output JSON to stdout (diagnostics go to stderr)
    eprintln!("  tokens ({} total): {:?}", tokens.len(), &tokens);
    print!("{{\"model_id\":\"{}\",\"dim\":{},\"tokens\":[", model_id, model.dim);
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{}", t);
    }
    print!("],\"embedding\":[");
    for (i, v) in embedding.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        // Full f32 precision
        print!("{:.10e}", v);
    }
    println!("]}}");
}
