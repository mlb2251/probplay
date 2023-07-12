using Gen


"""
Modified to remove MIME type stuff, to not only print the pretty verison when printing to console but instead always - bc there
are some cases where its failing to make it pretty even when printing
to console
"""
function Base.show(io::IO, choices::ChoiceMap)
    Gen._show_pretty(io, choices, 0, ())
end

function Base.show(io::IO, tr::Trace)
    show(io, get_choices(tr))
end

"""
Modified to abbreviate the printing of very long values (e.g. images) and to also print the scores
"""
function Gen._show_pretty(io::IO, choices::ChoiceMap, pre, vert_bars::Tuple)
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'
    indent_vert = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent_vert_last = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent = vcat(Char[' ' for _ in 1:pre], Char[PLUS, HORZ, HORZ, ' '])
    indent_last = vcat(Char[' ' for _ in 1:pre], Char[LAST, HORZ, HORZ, ' '])
    for i in vert_bars
        indent_vert[i] = VERT
        indent[i] = VERT
        indent_last[i] = VERT
    end
    indent_vert_str = join(indent_vert)
    indent_vert_last_str = join(indent_vert_last)
    indent_str = join(indent)
    indent_last_str = join(indent_last)
    key_and_values = collect(get_values_shallow(choices))
    key_and_submaps = collect(get_submaps_shallow(choices))
    n = length(key_and_values) + length(key_and_submaps)
    cur = 1
    for (key, value) in key_and_values
        # For strings, `print` is what we want; `Base.show` includes quote marks.
        # https://docs.julialang.org/en/v1/base/io-network/#Base.print
        print(io, indent_vert_str)

        ##################
        # MODIFIED START #
        ##################

        val_str = string(value)
        if length(val_str) > 50
            val_str = val_str[1:50] * "..."
        end

        score = get_val_score(choices, key)

        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $val_str [score=$score]\n")

        ################
        # MODIFIED END #
        ################

        cur += 1
    end
    for (key, submap) in key_and_submaps
        print(io, indent_vert_str)

        ##################
        # MODIFIED START #
        ##################

        score = get_submap_score(choices, key)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key))  [total score=$score] \n")

        ################
        # MODIFIED END #
        ################


        Gen._show_pretty(io, submap, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre+1))
        cur += 1

    end
end

function get_val_score(choices, key)
    "unable to print score for type: $(typeof(choices))"
end

function get_val_score(choices::Gen.DynamicDSLChoiceMap, key)
    choices.trie[key].score
end
function get_val_score(choices::Gen.StaticIRTraceAssmt, key)
    getfield(choices.trace, Symbol("$(Gen.choice_score_prefix)_$key"))
end

function get_submap_score(choices, key)
    "unable to print score for type: $(typeof(choices))"
end

function get_submap_score(choices::Gen.DynamicDSLChoiceMap, key)
    choices.trie[key].score
end
function get_submap_score(choices::Gen.StaticIRTraceAssmt, key)
    subtrace = getfield(choices.trace, Symbol("$(Gen.subtrace_prefix)_$key"))
    get_score(subtrace)
end

function get_submap_score(choices::Gen.VectorTraceChoiceMap, key::Int)
    get_score(choices.trace.subtraces[key])
end