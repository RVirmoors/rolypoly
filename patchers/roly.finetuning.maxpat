{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 5,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 179.0, 85.0, 748.0, 960.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"candycane" : 3,
					"contdata" : 1,
					"id" : "obj-9",
					"maxclass" : "multislider",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 340.0, 216.999995589256258, 80.0, 77.0 ],
					"setminmax" : [ 0.0, 2.0 ],
					"setstyle" : 1,
					"size" : 3
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 340.0, 298.666650767768942, 79.0, 22.0 ],
					"text" : "prepend train"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 274.0, 329.666655178512656, 145.0, 22.0 ],
					"presentation" : 1,
					"presentation_linecount" : 4,
					"presentation_rect" : [ 373.0, 280.0, 55.0, 64.0 ],
					"text" : "train 1.204678 0. 0.25731"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 9.5, 490.0, 34.0, 22.0 ],
					"text" : "write"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-85",
					"linecount" : 8,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 4.0, 71.0, 511.0, 126.0 ],
					"text" : "Finetuning the predictive model after a performance happens over 3 directions:\n- vel: attempts to match the generated velocities to those present in the score\n- off: matches the generated timing offsets to those performed by your instrument\n- gtr: adjusts vel&off generation to best predict your performed offsets\n\nThe first two are *adaptive* while the third is rather *predictive*. You can balance them by sending \"train $1 $2 $3\" messages to the object. Optimising across 1 or 2 directions may lead to more salient results compared to tracking all 3 at once.",
					"textcolor" : [ 0.426676, 0.426663, 0.42667, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Lato",
					"fontsize" : 48.0,
					"id" : "obj-6",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 4.0, 6.0, 472.0, 64.0 ],
					"text" : "rolypoly~"
				}

			}
, 			{
				"box" : 				{
					"args" : [ "@file", "jongly.aif", "@loop", 1, "@vol", 0 ],
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-2",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "demosound.maxpat",
					"numinlets" : 0,
					"numoutlets" : 1,
					"offset" : [ -4.0, -3.0 ],
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 479.0, 316.666667999999959, 219.0, 89.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"id" : "obj-28",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"name" : "roly.browse.maxpat",
					"numinlets" : 0,
					"numoutlets" : 0,
					"offset" : [ 0.0, 0.0 ],
					"patching_rect" : [ 575.0, 650.999999999999886, 123.0, 234.0 ],
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 2,
					"numoutlets" : 5,
					"orientation" : 1,
					"outlettype" : [ "signal", "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 104.0, 838.0, 136.0, 47.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_longname" : "live.gain~[3]",
							"parameter_mmax" : 6.0,
							"parameter_mmin" : -70.0,
							"parameter_shortname" : "drums",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "live.gain~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 104.0, 911.0, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 104.0, 636.567627000000016, 32.5, 23.0 ],
					"text" : "join"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-18",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 104.0, 606.567627000000016, 107.0, 23.0 ],
					"text" : "makenote 60 4n"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-11",
					"maxclass" : "newobj",
					"numinlets" : 7,
					"numoutlets" : 2,
					"outlettype" : [ "int", "" ],
					"patching_rect" : [ 104.0, 667.799996554851532, 108.0, 23.0 ],
					"text" : "midiformat"
				}

			}
, 			{
				"box" : 				{
					"autosave" : 1,
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"id" : "obj-12",
					"linecount" : 2,
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 8,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "signal", "signal", "", "list", "int", "", "", "" ],
					"patching_rect" : [ 104.0, 708.0, 300.0, 100.0 ],
					"save" : [ "#N", "vst~", "loaduniqueid", 0, "C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3", ";" ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_invisible" : 1,
							"parameter_longname" : "vst~",
							"parameter_shortname" : "vst~",
							"parameter_type" : 3
						}

					}
,
					"saved_object_attributes" : 					{
						"parameter_enable" : 1,
						"parameter_mappable" : 0
					}
,
					"snapshot" : 					{
						"filetype" : "C74Snapshot",
						"version" : 2,
						"minorversion" : 0,
						"name" : "snapshotlist",
						"origin" : "vst~",
						"type" : "list",
						"subtype" : "Undefined",
						"embed" : 1,
						"snapshot" : 						{
							"pluginname" : "LABS (64 Bit).vst3",
							"plugindisplayname" : "LABS",
							"pluginsavedname" : "C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3",
							"pluginsaveduniqueid" : 0,
							"version" : 1,
							"isbank" : 0,
							"isbase64" : 1,
							"sliderorder" : [  ],
							"slidervisibility" : [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
							"blob" : "9430.VMjLgzLI...OVMEUy.Ea0cVZtMEcgQWY9vSRC8Vav8lak4Fc9bCLwbiKV0zQicUPt3hKl4hKt3BTt3hKt3hKLoGVzMGQt3hbQQkQIoGTtEjKt3BR1QEa2oFVtPDTAEjKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKtnkZwU0PIMER58VPt3hc48zLvXTXlg0UYgWSWoUczX0SnQTZKYGRBgzZzDCV0EkUZQ2XV8DZTUTUFAiPNg1Mo8jY1MzTmkTLhkicSMUQQUETlgkUXM2ZFEFMvjFRDkzUiMWSsgjYyXEVyUkUOgFTpIFLvDiXn4hPhgGNFkELMYzXMgiQYsFLogjcHIDRwTEahk2ZwDFcvjFR2MiPLQGSogjYPcEVs0zUOgFQCwzctLDS14RZNQTRWM1bM0FRloWLgo1Zrk0aUYTV3fjTLg1Mn8zMTUkTlQ0UZk2ZrQ1ZvjFR2MiPLglKRM1aMESXxcmUXYWSWkkZvjFR2gDdKkicSAkTQUkTC0zZOcCSUEEUQUkTNMFQH8VTV8DZtHyU4sVagkVTvDFUUYUX1gCaHYFVWgkbUcUV3fjTLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWo1ZsE1YvXkVo0TaUs1cwDVZqYzXz.idgoVUrgjYXcEVxU0UYgCR3A0SvPDURUkdTMUUDEkYXUUTLgidPkTTUYkYlQkTGclZHU2LC8zTUQTUTslZScTPRokZvjFR1gCLh4FNrIldIUTUMgiQYsVRBgTLEYTXvTkUOgldnwDctjFR0MyPOMUUDUEUqo1TGEjTZoFLogjc3r1XqcWLgk1ZFMFMMQ0X3k0UYglKnM1Y2Y0XqASZHwzZpMUQEoFUlgUUQwDN5AURQUkUncCZOcCSUEEUQUkTNMFQH8VTV8DZtHyU2U0UXQWTWoUdUY0T0EkUYglKnM1Y2Y0XqASZHg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWM2ZFQ1ZIcETpk0UXQWSVkkZIIDRwTjQgASUV8DZDkFR0MyPOMUUDUEUqo1TGEjTZoFLogjc3TUXuc1UYg2XDEVcIYEVxkjPHESQFEFLUY0SnQTZHU2LC8zTUQTUTslZScTPRokZvjFR1gSUg81YWkEd2oWXoMGaHYFVWgkbUcUV3fjPLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWM2ZFQ1ZIcDUmMlUYglKnM1Y2Y0XqASZHg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWcVRGM1aMYzT00TLZglKnM1Y2Y0XqASZHYGR3sTN1kGUEEUQUkDM5EkYpYTV3fjPhUVQrIldqECVPUTLYsVRBgTLEYTXvTkUOgFR3sTN1kGUEEUQUkDM5EkYpYTV3fjPhU1cVgUdQICUqcmUYkVTWkkZAslXuAiUXg2ZWAEdQckVokjPHESQFEFLUY0Sn4RZHU2LC8zTUQTUTslZScTPRokZvjFR1gSUg8VTVo0PmYEVzQiUYIWRBgTLEYTXvTkUOglKogTcyLzSSUEQUQ0ZpM0QAIkVpASZHYGNvH1Z2YUVoE0UYoVTUgUaM0FRlg0UXIWUWkENHITT3U0UgkWR3sTN1kGUEEUQUkDM5EkYpYTV3fjPhUVVVokbQcUV3EDLgkWRBgTLEYTXvTkUOglbUcEZ2f1S2vTUQQUTUIkSiQDRuEkUOglKxbkcIcUV4UkQiAENwHFZtf1XmcmUisFLogTXAMzRl4xTWg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWsVVxD1P3vVX5UjUZQWUrIFT3DiXn4BZic1cVM1ZvjFRgAyZHU2LC8zTUQTUTslZScTPRokZvjFR1gCLgkVTWgULUwFRlg0UXIWUWkENHIDSncCZOcCSUEEUQUkTNMFQH8VTV8DZtHyU0EkUYkWTWIEcQYkVoUULhglKnM1Y2Y0XqASZHYmcRwjbHMzR4YmPMIGUCsTL1gWSxY1PKQicRwjc1IES2YmTLgmcRwTd1IES5YmTLAicRwTL1IESxXmTLMicRwDM1gFS1YGZLcmcnwDd1gFS4YGZLomcnwDL1gFSwXGZLIicnwzL1gFSzXGdLYmc3wzc1gGS3YGdLkmc3wjd1gGSvXGdLEic3wjL1gGSyXGdLQicB0jc1ITS2YmPMgmcB0Td1ITS5YmPMAicB0TL1ITSxXmPMMicB0DM1IUS1YmTMcmcR0Dd1IUS4YmTMomcR0DL1IUSwXmTMIicR0zL1IUSzXGZMYmcn0zc1gVS3YGZMkmcn0jd1gVSvXGZMEicn0jL1gVSyXGZMQic30jc1gWS2YGdMgmc30Td1gWS5YGdMAic30TL1gWSxXGdMMic30DM1IjS1YmPNcmcB4Dd1IjS4YmPNomcB4DL1IjSwXmPNIicB4zL1IjSzXmTNYmcR4zc1IkS3YmTNkmcR4jd1IkSvXmTNEicR4jL1IkSyXmTNQicRwjctLzR24xTLIGQCwDd1IES1wzPKcmKC0jbDMDSvXmTLYGVCszctjWSxQzPLMicRwjcpMzR2QzPLIGQSwzc1IES2gzPKcGQ4wjbDMES5YmTLcGUCszcDkVSxQzTLIicRwzclMzR2QzTNIGQowjc1IES3QzPKcGRowjbDkFS4YmTLgGTCszcHMUSxQTZLEicRwDdhMzR2gzPNIGQowDM1IES44xPKcGSSwjbDkGS3YmTLkGSCszcLMTSxQTdLAicRwTdXMzR2wTdMIGQ4wzL1IES4o1PKcGTCwjbDMTS2YmTLoGRCszcPkGSxQzPMomcRwjdTMzR2AUZMIGQC0jL1IES5Y1PKcGTS4jbDMUS1YmTLACQCszcTkFSxQzTMkmcRwDLPMzR2Q0TMIGQS0TL1IESvH1PKcGUC4jbDMUSzXmTLEiKCszcXMESxQTZMgmcRwTLLMzR2g0PMIGQo0DL1IESwf0PKcGV40jbDkVSyXmTLEiZCszchMDSxQTdMcmcRwjLHMzR2IVdLIGQ40jd1IESxP0PKcmXo0jbDkWSxXmTLIiYCszchMkSxQzPNYmcRwzLDMzR2YVZLIGQC4Td1IESy.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.."
						}
,
						"snapshotlist" : 						{
							"current_snapshot" : 0,
							"entries" : [ 								{
									"filetype" : "C74Snapshot",
									"version" : 2,
									"minorversion" : 0,
									"name" : "LABS",
									"origin" : "LABS (64 Bit).vst3",
									"type" : "VST3",
									"subtype" : "Instrument",
									"embed" : 0,
									"snapshot" : 									{
										"pluginname" : "LABS (64 Bit).vst3",
										"plugindisplayname" : "LABS",
										"pluginsavedname" : "C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3",
										"pluginsaveduniqueid" : 0,
										"version" : 1,
										"isbank" : 0,
										"isbase64" : 1,
										"sliderorder" : [  ],
										"slidervisibility" : [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
										"blob" : "9430.VMjLgzLI...OVMEUy.Ea0cVZtMEcgQWY9vSRC8Vav8lak4Fc9bCLwbiKV0zQicUPt3hKl4hKt3BTt3hKt3hKLoGVzMGQt3hbQQkQIoGTtEjKt3BR1QEa2oFVtPDTAEjKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKt3hKtnkZwU0PIMER58VPt3hc48zLvXTXlg0UYgWSWoUczX0SnQTZKYGRBgzZzDCV0EkUZQ2XV8DZTUTUFAiPNg1Mo8jY1MzTmkTLhkicSMUQQUETlgkUXM2ZFEFMvjFRDkzUiMWSsgjYyXEVyUkUOgFTpIFLvDiXn4hPhgGNFkELMYzXMgiQYsFLogjcHIDRwTEahk2ZwDFcvjFR2MiPLQGSogjYPcEVs0zUOgFQCwzctLDS14RZNQTRWM1bM0FRloWLgo1Zrk0aUYTV3fjTLg1Mn8zMTUkTlQ0UZk2ZrQ1ZvjFR2MiPLglKRM1aMESXxcmUXYWSWkkZvjFR2gDdKkicSAkTQUkTC0zZOcCSUEEUQUkTNMFQH8VTV8DZtHyU4sVagkVTvDFUUYUX1gCaHYFVWgkbUcUV3fjTLg1Mn8zMLUUTTEUUR4zXDgzaQY0Sn4hLWo1ZsE1YvXkVo0TaUs1cwDVZqYzXz.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.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.."
									}
,
									"fileref" : 									{
										"name" : "LABS",
										"filename" : "LABS.maxsnap",
										"filepath" : "~/Documents/Max 8/Snapshots",
										"filepos" : -1,
										"snapshotfileid" : "d39badcbaa16975273137bb8d1553906"
									}

								}
 ]
						}

					}
,
					"text" : "vst~ \"C:/Program Files/Common Files/VST3/LABS (64 Bit).vst3\"",
					"varname" : "vst~",
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-103",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 350.0, 546.166667999999959, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-101",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "bang", "bang" ],
					"patching_rect" : [ 350.0, 514.666667999999959, 42.0, 22.0 ],
					"text" : "edge~"
				}

			}
, 			{
				"box" : 				{
					"channels" : 1,
					"id" : "obj-100",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 1,
					"numoutlets" : 4,
					"orientation" : 1,
					"outlettype" : [ "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 479.0, 417.666667999999959, 136.0, 41.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_longname" : "live.gain~[2]",
							"parameter_mmax" : 6.0,
							"parameter_mmin" : -70.0,
							"parameter_shortname" : "guitar",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "live.gain~[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-98",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 452.0, 471.5, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-55",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 9.5, 458.5, 32.0, 22.0 ],
					"text" : "train"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 13.0,
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "" ],
					"patching_rect" : [ 163.0, 396.999995589256287, 90.0, 23.0 ],
					"text" : "getattr latency"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 13.0,
					"id" : "obj-17",
					"linecount" : 2,
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "" ],
					"patching_rect" : [ 198.5, 442.0, 174.5, 38.0 ],
					"text" : "fluid.onsetslice~ @metric 9 @threshold 0.22"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-52",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 9.5, 268.333323178512615, 33.0, 22.0 ],
					"text" : "read"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 9.5, 294.999995589256287, 32.0, 22.0 ],
					"text" : "start"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 11,
					"outlettype" : [ "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "", "" ],
					"patching_rect" : [ 9.5, 549.5, 124.0, 22.0 ],
					"text" : "rolypoly~ tim.mid"
				}

			}
, 			{
				"box" : 				{
					"attr" : "latency",
					"id" : "obj-57",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 163.0, 514.666667999999959, 150.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"attr" : "filter_hits",
					"id" : "obj-4",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 59.5, 238.999995589256287, 225.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"attr" : "out_message",
					"id" : "obj-5",
					"maxclass" : "attrui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 59.5, 212.999995589256258, 150.0, 22.0 ]
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 0 ],
					"source" : [ "obj-1", 9 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 1 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 1 ],
					"source" : [ "obj-100", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-103", 0 ],
					"source" : [ "obj-101", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-11", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 1 ],
					"source" : [ "obj-12", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"source" : [ "obj-13", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-57", 0 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 283.5, 367.0, 19.0, 367.0, 19.0, 536.0, 19.0, 536.0 ],
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"order" : 1,
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-101", 0 ],
					"order" : 0,
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 1 ],
					"midpoints" : [ 201.5, 632.567627000000016, 127.0, 632.567627000000016 ],
					"source" : [ "obj-18", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-16", 0 ],
					"midpoints" : [ 113.5, 631.567627000000016, 113.5, 631.567627000000016 ],
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-100", 0 ],
					"order" : 0,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"order" : 1,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 1 ],
					"source" : [ "obj-30", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-24", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 69.0, 445.0, 19.0, 445.0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 69.0, 235.0, 69.0, 235.0, 69.0, 445.0, 19.0, 445.0 ],
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-52", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-55", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 172.5, 543.0, 19.0, 543.0 ],
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"source" : [ "obj-9", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-100" : [ "live.gain~[2]", "guitar", 0 ],
			"obj-12" : [ "vst~", "vst~", 0 ],
			"obj-2::obj-21::obj-6" : [ "live.tab[3]", "live.tab[1]", 0 ],
			"obj-2::obj-35" : [ "[5]", "Level", 0 ],
			"obj-30" : [ "live.gain~[3]", "drums", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "LABS.maxsnap",
				"bootpath" : "~/Documents/Max 8/Snapshots",
				"patcherrelativepath" : "../../../Snapshots",
				"type" : "mx@s",
				"implicit" : 1
			}
, 			{
				"name" : "demosound.maxpat",
				"bootpath" : "C74:/help/msp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "fluid.onsetslice~.mxe64",
				"type" : "mx64"
			}
, 			{
				"name" : "interfacecolor.js",
				"bootpath" : "C74:/interfaces",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "random.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "roly.browse.maxpat",
				"bootpath" : "~/Documents/Max 8/Packages/rolypoly/patchers",
				"patcherrelativepath" : ".",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rolypoly~.mxe64",
				"type" : "mx64"
			}
, 			{
				"name" : "saw.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "sine.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
, 			{
				"name" : "square.svg",
				"bootpath" : "C74:/media/max/picts/m4l-picts",
				"type" : "svg",
				"implicit" : 1
			}
 ],
		"autosave" : 0,
		"styles" : [ 			{
				"name" : "rnbolight",
				"default" : 				{
					"accentcolor" : [ 0.443137254901961, 0.505882352941176, 0.556862745098039, 1.0 ],
					"bgcolor" : [ 0.796078431372549, 0.862745098039216, 0.925490196078431, 1.0 ],
					"bgfillcolor" : 					{
						"angle" : 270.0,
						"autogradient" : 0.0,
						"color" : [ 0.835294117647059, 0.901960784313726, 0.964705882352941, 1.0 ],
						"color1" : [ 0.031372549019608, 0.125490196078431, 0.211764705882353, 1.0 ],
						"color2" : [ 0.263682, 0.004541, 0.038797, 1.0 ],
						"proportion" : 0.39,
						"type" : "color"
					}
,
					"clearcolor" : [ 0.898039, 0.898039, 0.898039, 1.0 ],
					"color" : [ 0.815686274509804, 0.509803921568627, 0.262745098039216, 1.0 ],
					"editing_bgcolor" : [ 0.898039, 0.898039, 0.898039, 1.0 ],
					"elementcolor" : [ 0.337254901960784, 0.384313725490196, 0.462745098039216, 1.0 ],
					"fontname" : [ "Lato" ],
					"locked_bgcolor" : [ 0.898039, 0.898039, 0.898039, 1.0 ],
					"stripecolor" : [ 0.309803921568627, 0.698039215686274, 0.764705882352941, 1.0 ],
					"textcolor_inverse" : [ 0.0, 0.0, 0.0, 1.0 ]
				}
,
				"parentstyle" : "",
				"multi" : 0
			}
 ]
	}

}
